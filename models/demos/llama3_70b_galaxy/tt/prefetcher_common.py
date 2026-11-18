# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
import os

import ttnn
import torch
from loguru import logger
from models.common.lightweightmodule import LightweightModule

from models.demos.llama3_70b_galaxy.tt.model_config import (
    get_core_ranges,
)

global_tt_tensor_address = None
# Blackhole: top address of the first decode global CB, per mesh device. Every rebuild is placed back here.
_bh_global_cb_top = {}


def _mesh_is_blackhole(mesh_device):
    """Best-effort Blackhole detection from the mesh device arch."""
    try:
        arch = mesh_device.arch()
        return "blackhole" in str(arch).lower()
    except Exception:
        try:
            return "blackhole" in ttnn.get_arch_name().lower()
        except Exception:
            return False


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
        # Blackhole galaxy has 8 DRAM banks and a 12x10 tensix grid (vs Wormhole's 12 banks / 7x10),
        # so the prefetcher ring is 8 readers x 3 receivers (=24, same RING_SIZE) instead of 12 x 2.
        is_blackhole = _mesh_is_blackhole(mesh_device)
        self.is_blackhole = is_blackhole
        if is_blackhole:
            num_reader_cores = 8
            num_global_cb_receivers = 3
        else:
            num_reader_cores = 12
            num_global_cb_receivers = 2

        (
            self.active_sender_cores,
            self.dram_cores,
            self.all_sender_cores,
            self.active_receiver_cores_list,
            self.all_receiver_cores,
            self.worker_cores_range_set,
            self.mm_optimised_ring_cores,
            self.hop_grid,
        ) = get_core_ranges(
            num_reader_cores, num_global_cb_receivers, is_functional_test=False, is_blackhole=is_blackhole
        )

        ##### Set up the input tensors #####
        self.dram_core_range_set = ttnn.CoreRangeSet(
            [ttnn.CoreRange(core_coord, core_coord) for core_coord in self.dram_cores]
        )
        self.sender_core_range_set = ttnn.CoreRangeSet(
            [ttnn.CoreRange(core_coord, core_coord) for core_coord in self.active_sender_cores]
        )

        _grid_max = ttnn.CoreCoord(11, 9) if is_blackhole else ttnn.CoreCoord(6, 9)
        self.all_core_range_set = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), _grid_max)])

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
            # Blackhole: the global CB is top-anchored in L1 (its top sits just below the receiver-core
            # semaphore buffers and it grows downward). At 728 tiles its base lands at ~389 KB on the
            # prefetcher receiver cores, which leaves only ~389 KB of contiguous low L1 for the ring
            # matmul's static CBs (`bmm_..._gathered`, which need ~444 KB) -> during full-model trace
            # capture (global CB + every matmul program resident at once) the matmul CBs overlap the
            # global CB by ~55 KB ("static circular buffers clash with L1 buffers on [1-0 - 6-9]").
            # Trimming to 656 tiles raises the global CB base to ~457 KB, clearing the matmul CB region
            # with margin across all receiver cores. Still >2x the largest single weight block, so the
            # weight double-buffering the prefetcher relies on is preserved. WH (2 receivers, different
            # L1 layout) keeps the original 728.
            # LLAMA_BH_GLOBAL_CB_TILES: on BH a smaller global CB moves its (top-anchored) base above the prefill
            # programs' static CBs, so prefill can run while the decode CB is alive (vLLM switches every request).
            bh_tiles = int(os.environ.get("LLAMA_BH_GLOBAL_CB_TILES", "656"))
            self.global_cb_size = (bh_tiles if self.is_blackhole else 728) * 1088
            self.sender_receiver_mapping = list(zip(self.all_sender_cores, self.all_receiver_cores))
            # self.global_circular_buffer = ttnn.create_global_circular_buffer(
            #     self.mesh_device, self.sender_receiver_mapping, self.global_cb_size
            # )
            # logger.info(f"GlobalCB size {self.global_cb_size}")
            self.global_circular_buffer = None  # Global CB will only be allocated before decode runs
            self.prefetcher_sub_device = ttnn.SubDevice([self.sender_core_range_set])
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
            if self.is_blackhole and os.environ.get("LLAMA_BH_PIN_CB", "1") == "1":
                self._create_global_cb_bh()
            else:
                self.global_circular_buffer = ttnn.create_global_circular_buffer(
                    self.mesh_device,
                    self.sender_receiver_mapping,
                    self.global_cb_size,
                )

    def _l1_pad(self, nbytes):
        """Interleaved L1 buffer of nbytes on every L1 bank; top-down allocation puts it at the top of the highest free block."""
        num_banks = ttnn.get_memory_view(self.mesh_device, ttnn.BufferType.L1).num_banks
        return ttnn.from_torch(
            torch.zeros(1, 1, num_banks, nbytes // 4, dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

    def _global_cb_top(self):
        tops = [
            int(b["address"]) + int(b["size"])
            for b in map(dict, ttnn.get_memory_view(self.mesh_device, ttnn.BufferType.L1).block_table)
            if str(b.get("allocated", "")).lower() in ("yes", "true", "1") and int(b["size"]) == self.global_cb_size
        ]
        return max(tops) if tops else None

    def _create_global_cb_bh(self):
        # Decode ops allocate small L1 buffers on first compile that the program cache keeps for good. Top-down
        # allocation puts them just below the global CB, so once the CB is freed for prefill they sit in the middle
        # of L1 and clash with prefill's static CBs (which need contiguous low L1, ~1.09 MB for 2K SDPA). The first
        # CB is therefore created under a free gap that takes those buffers, and every rebuild is pinned to the
        # same top address (the decode trace bakes the CB address in) by padding the free space above it.
        key = id(self.mesh_device)
        top = _bh_global_cb_top.get(key)
        pads = []
        if top is None:
            gap = int(os.environ.get("LLAMA_BH_GLOBAL_CB_GAP_BYTES", "65536"))
            if gap > 0:
                pads.append(self._l1_pad(gap))
        else:
            # Fill every free hole above the target, highest first, so the CB's top lands exactly at `top`.
            free = []
            for block in map(dict, ttnn.get_memory_view(self.mesh_device, ttnn.BufferType.L1).block_table):
                if str(block.get("allocated", "")).lower() in ("yes", "true", "1"):
                    continue
                start = int(block["address"])
                free.append((start, start + int(block["size"])))
            holder = [(a, b) for a, b in free if a <= top - self.global_cb_size and b >= top]
            if not holder:
                logger.warning(f"Blackhole global CB: no free L1 block holds [{top - self.global_cb_size}, {top})")
            for a, b in sorted(free, reverse=True):
                if a >= top:
                    pads.append(self._l1_pad(b - a))
            if holder and holder[0][1] > top:
                pads.append(self._l1_pad(holder[0][1] - top))
        self.global_circular_buffer = ttnn.create_global_circular_buffer(
            self.mesh_device,
            self.sender_receiver_mapping,
            self.global_cb_size,
        )
        for pad in pads:
            ttnn.deallocate(pad)
        cb_top = self._global_cb_top()
        if top is None:
            _bh_global_cb_top[key] = cb_top
            logger.info(f"Blackhole global CB top pinned at {cb_top}")
        elif cb_top != top:
            logger.warning(f"Blackhole global CB moved: top {cb_top}, expected {top}")

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
