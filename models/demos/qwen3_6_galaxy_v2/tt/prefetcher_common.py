# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import torch
from loguru import logger

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.utility_functions import is_blackhole
from models.demos.llama3_70b_galaxy.tt.model_config import get_core_ranges

global_tt_tensor_address = None


def get_bh_prefetcher_core_ranges(num_global_cb_receivers=2):
    """Blackhole 8-DRAM-bank prefetcher core layout.

    Returns the same 8-tuple shape that ``llama3_70b_galaxy.get_core_ranges``
    returns (so ``TtLlamaPrefetcherSetup`` can unpack it identically), but built
    from the BH-correct config in ``tt_transformers/tt/prefetcher.py``
    (``PrefetcherCoreConfig`` + ``prefetcher_config.yaml``) instead of the
    Wormhole 12-bank hardcode. BH: 8 banks at X=[1,3,2,0,5,7,6,4], senders on
    cols 0/7, receivers in cols 1-7 / 8-11.

    CPU-only (PrefetcherCoreConfig uses only the yaml cfg, not the device).
    ``num_global_cb_receivers`` <= 3 uses the default (no override) layout.

    NOTE: ``mm_optimised_ring_cores`` / ``hop_grid`` are returned empty here —
    they configure the prefetched-weight ring MATMUL (G2+), not the global_cb
    allocation (G1). They must be populated (from the receiver grid) before the
    decode ring matmul runs through the prefetcher.
    """
    from models.tt_transformers.tt.prefetcher import ARCH_CONFIG, PrefetcherCoreConfig

    cfg = PrefetcherCoreConfig(
        num_receiver_cores=num_global_cb_receivers,
        mesh_device=None,  # __post_init__ only reads the yaml cfg
        cfg=ARCH_CONFIG["blackhole"],
    )

    dram_cores = cfg.dram_banks()  # 8 CoreCoord at row 0
    active_sender_cores = cfg.sender_cores(active=True)  # 8 CoreCoord on cols 0/7
    all_sender_cores = list(active_sender_cores)
    # One CoreRangeSet of receivers per (active) sender — uniform size, as the
    # global circular buffer requires.
    all_receiver_cores = cfg.receiver_cores(sender_active=True, receiver_active=True)

    # Flat (x,y) receiver list in sender order (used downstream to build the
    # matmul output mem-config grid; pf_receiver_cores_list).
    active_receiver_cores_list = []
    for crs in all_receiver_cores:
        for cr in crs.ranges():
            for y in range(cr.start.y, cr.end.y + 1):
                for x in range(cr.start.x, cr.end.x + 1):
                    active_receiver_cores_list.append((x, y))

    # Worker sub-device grid (prefetcher_common overrides this for BH anyway):
    # everything except the sender columns 0 and 7.
    worker_cores_range_set = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(6, 9)),
            ttnn.CoreRange(ttnn.CoreCoord(8, 0), ttnn.CoreCoord(11, 9)),
        ]
    )

    mm_optimised_ring_cores = []  # G2: populate from receiver grid for the ring matmul
    hop_grid = []

    return (
        active_sender_cores,
        dram_cores,
        all_sender_cores,
        active_receiver_cores_list,
        all_receiver_cores,
        worker_cores_range_set,
        mm_optimised_ring_cores,
        hop_grid,
    )


class TtLlamaPrefetcherSetup(LightweightModule):
    def __init__(
        self,
        mesh_device,
        n_tensors,
        n_layers,
        mode="decode",
        mesh_sub_device_manager_id_prefill=None,
        mesh_sub_device_manager_id_decode=None,
        save_tensor_addresses=False,
        is_qwen=False,
    ):
        """
        - sub devices
        - global cb
        - helper functions to get the weight addresses
        """
        logger.info("Running TtLlamaPrefetcherSetup")

        self.mesh_device = mesh_device
        self.n_tensors = n_tensors
        self.n_layers = n_layers

        ###### Set up GlobalCB ######
        num_global_cb_receivers = 2

        if is_blackhole():
            # BH GLX: 8 DRAM banks at X=[1,3,2,0,5,7,6,4], senders cols 0/7. The
            # WH get_core_ranges hardcodes 12 banks -> "bank x=8" on BH.
            (
                self.active_sender_cores,
                self.dram_cores,
                self.all_sender_cores,
                self.active_receiver_cores_list,
                self.all_receiver_cores,
                self.worker_cores_range_set,
                self.mm_optimised_ring_cores,
                self.hop_grid,
            ) = get_bh_prefetcher_core_ranges(num_global_cb_receivers=num_global_cb_receivers)
        else:
            num_reader_cores = 12
            (
                self.active_sender_cores,
                self.dram_cores,
                self.all_sender_cores,
                self.active_receiver_cores_list,
                self.all_receiver_cores,
                self.worker_cores_range_set,
                self.mm_optimised_ring_cores,
                self.hop_grid,
            ) = get_core_ranges(num_reader_cores, num_global_cb_receivers, is_functional_test=False)

        ##### Set up the input tensors #####
        self.dram_core_range_set = ttnn.CoreRangeSet(
            [ttnn.CoreRange(core_coord, core_coord) for core_coord in self.dram_cores]
        )
        self.sender_core_range_set = ttnn.CoreRangeSet(
            [ttnn.CoreRange(core_coord, core_coord) for core_coord in self.active_sender_cores]
        )

        # Determine actual compute grid from device; WH TG: (7,10), BH GLX: (13,10)
        grid_size = mesh_device.compute_with_storage_grid_size()
        # Full compute grid covering all Tensix cores on this device type
        self.all_core_range_set = ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_size.x - 1, grid_size.y - 1))]
        )

        ##### Setup up sub devices #####

        if mode == "prefill":
            self.all_sub_device = ttnn.SubDevice([self.all_core_range_set])
            self.all_sub_device_id = ttnn.SubDeviceId(0)
            self.worker_sub_device_id = self.all_sub_device_id
            if mesh_sub_device_manager_id_prefill is None:
                mesh_sub_device_manager_id_prefill = mesh_device.create_sub_device_manager([self.all_sub_device], 0)
            self.mesh_sub_device_manager_id_prefill = mesh_sub_device_manager_id_prefill
            mesh_device.load_sub_device_manager(self.mesh_sub_device_manager_id_prefill)
            mesh_device.set_sub_device_stall_group([self.worker_sub_device_id])
        else:
            ##### Set up the global circular buffer #####
            # Global CB must be large enough to atleast double buffer weights
            # This ensures that back to back matmuls (for eg. in MLP) can run
            # without stalling on the weight prefetch
            # To fit entire MLP we'd need ~742 * 1088 but using block-wise prefetching and 732 tiles this is sufficient for now
            self.global_cb_size = 728 * 1088
            self.sender_receiver_mapping = list(zip(self.all_sender_cores, self.all_receiver_cores))
            # self.global_circular_buffer = ttnn.create_global_circular_buffer(
            #     self.mesh_device, self.sender_receiver_mapping, self.global_cb_size
            # )
            # logger.info(f"GlobalCB size {self.global_cb_size}")
            self.global_circular_buffer = None  # Global CB will only be allocated before decode runs
            self.prefetcher_sub_device = ttnn.SubDevice([self.sender_core_range_set])
            # On BH (13x10 grid), extend worker cores to include cols 7-12 in addition to WH's cols 1-3, 5-6.
            # WH uses cols 0 and 4 for prefetcher senders; those stay in prefetcher_sub_device on BH too.
            if is_blackhole():
                bh_worker_cores_range_set = ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, grid_size.y - 1)),
                        ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(grid_size.x - 1, grid_size.y - 1)),
                    ]
                )
                self.worker_sub_device = ttnn.SubDevice([bh_worker_cores_range_set])
            else:
                self.worker_sub_device = ttnn.SubDevice([self.worker_cores_range_set])
            self.prefetcher_sub_device_id = ttnn.SubDeviceId(0)
            self.worker_sub_device_id = ttnn.SubDeviceId(1)
            if mesh_sub_device_manager_id_decode is None:
                mesh_sub_device_manager_id_decode = mesh_device.create_sub_device_manager(
                    [self.prefetcher_sub_device, self.worker_sub_device], 0
                )
            self.mesh_sub_device_manager_id_decode = mesh_sub_device_manager_id_decode
            mesh_device.load_sub_device_manager(self.mesh_sub_device_manager_id_decode)
            mesh_device.set_sub_device_stall_group([self.prefetcher_sub_device_id, self.worker_sub_device_id])

        self.tensors = []
        self.tensor_addrs = []  # List of buffer addresses
        self.save_tensor_addresses = save_tensor_addresses

    def create_global_cb(self):
        if not hasattr(self, "global_circular_buffer") or self.global_circular_buffer is None:
            self.global_circular_buffer = ttnn.create_global_circular_buffer(
                self.mesh_device,
                self.sender_receiver_mapping,
                self.global_cb_size,
            )

    def insert_tensor(self, tensor: ttnn.Tensor):
        self.tensors.append(tensor)
        self.tensor_addrs.append(tensor.buffer_address())

    def get_tensor_addrs(self):
        assert (
            len(self.tensor_addrs) == self.n_tensors * self.n_layers
        ), f"Expected {self.n_tensors * self.n_layers} tensor addresses, got {len(self.tensor_addrs)}"

        tensor_addrs = torch.tensor(self.tensor_addrs)
        tensor_addrs = tensor_addrs.repeat(len(self.dram_cores), 1)
        tensor_addrs_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                self.sender_core_range_set,
                [tensor_addrs.shape[0] // len(self.dram_cores), tensor_addrs.shape[1]],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )
        tt_tensor_addrs = ttnn.as_tensor(
            tensor_addrs,
            device=self.mesh_device,
            dtype=ttnn.uint32,
            memory_config=tensor_addrs_mem_config,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

        return tt_tensor_addrs

    def get_input_tensors(self):
        assert (
            len(self.tensors) >= self.n_tensors
        ), f"Expected at least {self.n_tensors} tensors, got {len(self.tensors)}"
        if self.save_tensor_addresses:
            global global_tt_tensor_address
            if global_tt_tensor_address is None:
                global_tt_tensor_address = self.get_tensor_addrs()
        else:
            global_tt_tensor_address = self.get_tensor_addrs()
        self.tt_tensor_address = global_tt_tensor_address
        return self.tensors[: self.n_tensors] + [self.tt_tensor_address]
