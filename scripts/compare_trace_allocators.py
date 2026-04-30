# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Compare trace allocation heuristics on a captured trace allocation JSON file.

The input format is the schema emitted by the trace allocation capture hook:
core type metadata, ringbuffer configs, trace nodes, and optionally the captured
SimpleTraceAllocator results.  This script replays two allocators:

* SimpleTraceAllocator, ported from tt_metal/impl/dispatch/simple_trace_allocator.cpp
* trace-cache2, ported from pkeller/trace-cache2:tt_metal/impl/dispatch/trace_cache.cc

Badness is computed as requested: for a stall at trace node i that waits for
trace node j, stall_distance = i - j - 1, and contribution = 1 / 2^stall_distance.
Stalls at least launch_msg_buffer_num_entries - 1 trace nodes back contribute
zero badness because they fit in the launch message window.  Synthetic
first-program stalls are not included because they do not stall on a previous
trace program.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


LAUNCH_MSG_BUFFER_NUM_ENTRIES = 8
SIMPLE_NONBINARY = 0
SIMPLE_BINARY = 1
SIMPLE_NUM_DATA_TYPES = 2
TRACE_CACHE_MAX_REUSE_WINDOW = 2


@dataclass
class CoreTypeInfo:
    index: int
    core_type: str
    binary_in_config: bool
    has_separate_binary_offset: bool
    skip: bool


@dataclass
class RingbufferConfig:
    start: int
    size: int


@dataclass
class PerCoreTypeNode:
    index: int
    nonbinary_size: int
    binary_size: int
    has_kernel_groups: bool


@dataclass
class TraceNodeInput:
    program_id: int
    sub_device_id: int
    num_workers: int
    per_core_type: dict[int, PerCoreTypeNode]


@dataclass
class ExtraData:
    next_use_idx: list[int | None] = field(default_factory=lambda: [None] * SIMPLE_NUM_DATA_TYPES)
    finished_sync_count: int = 0


@dataclass
class MemoryUsage:
    trace_idx: int
    data_type: int
    size: int
    program_id: int


@dataclass
class SimpleMetadata:
    binary_addrs: list[int]
    nonbinary_addrs: list[int]
    send_binary: bool = False
    stall_before_program: bool = False
    stall_first: bool = False
    sync_count: int = 0
    stall_idx: int | None = None
    dispatch_bytes: int = 0


@dataclass(eq=False)
class TraceCacheTraceNode:
    program_id: int
    remaining: int = 0
    next_idx: int | None = None
    weight: float = 0.0
    addr: int | None = None
    does_dispatch: bool = False
    stall_idx: int | None = None


@dataclass
class TraceCacheProgram:
    size: int
    cost: int


@dataclass(eq=False)
class TraceCacheAllocNode:
    program_id: int
    addr: int
    size: int
    first_use: int
    prev_use: int
    is_free: bool = False

    @classmethod
    def free(cls, addr: int, size: int) -> TraceCacheAllocNode:
        return cls(program_id=-1, addr=addr, size=size, first_use=-TRACE_CACHE_MAX_REUSE_WINDOW, prev_use=-TRACE_CACHE_MAX_REUSE_WINDOW, is_free=True)

    def weight(self, trace: list[TraceCacheTraceNode]) -> float:
        if self.is_free:
            return 0.0
        return trace[self.prev_use].weight


@dataclass
class AllocatorStats:
    name: str
    nodes: int
    stalls: int
    badness: float
    immediate_previous_stalls: int
    min_stall_distance: int | None
    max_stall_distance: int | None
    dispatch_bytes: int


def intersects(begin_1: int, size_1: int, begin_2: int, size_2: int) -> bool:
    return begin_1 < begin_2 + size_2 and begin_2 < begin_1 + size_1


def merge_syncs(sync_1: int | None, sync_2: int | None) -> int | None:
    if sync_1 is not None and sync_2 is not None:
        return max(sync_1, sync_2)
    if sync_1 is not None:
        return sync_1
    return sync_2


class SimpleRegionAllocator:
    def __init__(self, ringbuffer_size: int, extra_data: list[ExtraData]) -> None:
        self.ringbuffer_size = ringbuffer_size
        self.extra_data = extra_data
        self.program_ids_memory_map: list[dict[int, int]] = [dict() for _ in range(SIMPLE_NUM_DATA_TYPES)]
        self.regions: dict[int, MemoryUsage] = {}

    def reset_allocator(self) -> None:
        for memory_map in self.program_ids_memory_map:
            memory_map.clear()
        self.regions.clear()

    def add_region(self, data_type: int, program_id: int, addr: int) -> None:
        self.program_ids_memory_map[data_type][program_id] = addr

    def get_region(self, data_type: int, program_id: int) -> int | None:
        return self.program_ids_memory_map[data_type].get(program_id)

    def update_region_trace_idx(self, region_addr: int, trace_idx: int) -> None:
        if region_addr in self.regions:
            self.regions[region_addr].trace_idx = trace_idx

    def allocate_region(
        self, size: int, trace_idx: int, data_type: int, program_id: int
    ) -> tuple[int | None, int | None]:
        if size == 0:
            return None, 0

        best_addr: int | None = None
        best_cost = math.inf
        best_region_sync_idx: int | None = None
        marked_for_deletion: set[int] = set()

        # Same candidates as the C++ allocator: zero and the end of every
        # existing region.  Regions are non-overlapping after each allocation.
        sorted_regions = sorted(self.regions.items())
        candidate_addrs = [0] + [addr + usage.size for addr, usage in sorted_regions]

        for addr in candidate_addrs:
            if addr + size > self.ringbuffer_size:
                break

            cost = 0.0
            region_sync_idx: int | None = None
            now_in_use = False

            for region_addr, region in sorted_regions:
                if region_addr >= addr + size:
                    break
                if not intersects(addr, size, region_addr, region.size):
                    continue

                if region.trace_idx == trace_idx:
                    now_in_use = True
                    break

                next_use_idx = self.extra_data[region.trace_idx].next_use_idx[region.data_type]
                if next_use_idx is not None:
                    if next_use_idx == trace_idx:
                        cost += 1_000_000_000
                    else:
                        cost += region.size * 1.0 / (next_use_idx - trace_idx)
                elif trace_idx - region.trace_idx > LAUNCH_MSG_BUFFER_NUM_ENTRIES:
                    marked_for_deletion.add(region_addr)

                region_sync_idx = merge_syncs(region_sync_idx, region.trace_idx)

            if now_in_use:
                continue

            if region_sync_idx is not None:
                desired_write_ahead = min(LAUNCH_MSG_BUFFER_NUM_ENTRIES, 7)
                region_idx_diff = trace_idx - region_sync_idx
                if region_idx_diff < desired_write_ahead:
                    cost += 100_000_000 * (1 << (desired_write_ahead - region_idx_diff))

            if cost < best_cost:
                best_cost = cost
                best_addr = addr
                best_region_sync_idx = region_sync_idx
            if cost == 0:
                break

        for addr in marked_for_deletion:
            region = self.regions.get(addr)
            if region is None:
                continue
            self.program_ids_memory_map[region.data_type].pop(region.program_id, None)
            del self.regions[addr]

        if best_addr is None:
            return None, None

        for addr, region in list(self.regions.items()):
            if intersects(best_addr, size, addr, region.size):
                self.program_ids_memory_map[region.data_type].pop(region.program_id, None)
                del self.regions[addr]

        self.regions[best_addr] = MemoryUsage(trace_idx, data_type, size, program_id)
        return best_region_sync_idx, best_addr


def load_capture(path: Path) -> tuple[list[CoreTypeInfo], list[RingbufferConfig], list[TraceNodeInput], list[dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    core_types = [
        CoreTypeInfo(
            index=core["index"],
            core_type=core["core_type"],
            binary_in_config=core["binary_in_config"],
            has_separate_binary_offset=core["has_separate_binary_offset"],
            skip=core["skip"],
        )
        for core in data["core_types"]
    ]
    ringbuffer_configs = [
        RingbufferConfig(start=config["start"], size=config["size"]) for config in data["ringbuffer_configs"]
    ]
    trace_nodes = []
    for node in data["trace_nodes"]:
        per_core_type = {
            entry["index"]: PerCoreTypeNode(
                index=entry["index"],
                nonbinary_size=entry["nonbinary_size"],
                binary_size=entry["binary_size"],
                has_kernel_groups=entry["has_kernel_groups"],
            )
            for entry in node["per_core_type"]
        }
        trace_nodes.append(
            TraceNodeInput(
                program_id=node["program_id"],
                sub_device_id=node["sub_device_id"],
                num_workers=node["num_workers"],
                per_core_type=per_core_type,
            )
        )

    return core_types, ringbuffer_configs, trace_nodes, data.get("results", [])


def build_simple_extra_data(trace_nodes: list[TraceNodeInput]) -> list[ExtraData]:
    extra_data = [ExtraData() for _ in trace_nodes]
    program_ids_use_map: dict[int, int] = {}
    for i in range(len(trace_nodes) - 1, -1, -1):
        program_id = trace_nodes[i].program_id
        if program_id in program_ids_use_map:
            extra_data[i].next_use_idx[SIMPLE_BINARY] = program_ids_use_map[program_id]
        program_ids_use_map[program_id] = i
    return extra_data


def run_simple_allocator(
    core_types: list[CoreTypeInfo], ringbuffer_configs: list[RingbufferConfig], trace_nodes: list[TraceNodeInput]
) -> list[SimpleMetadata]:
    extra_data = build_simple_extra_data(trace_nodes)
    region_allocators = [SimpleRegionAllocator(config.size, extra_data) for config in ringbuffer_configs]
    metadata = [
        SimpleMetadata(binary_addrs=[0] * len(core_types), nonbinary_addrs=[0] * len(core_types))
        for _ in trace_nodes
    ]

    sub_device_ids = sorted({node.sub_device_id for node in trace_nodes})
    for sub_device_id in sub_device_ids:
        for allocator in region_allocators:
            allocator.reset_allocator()

        expected_workers_completed = 0
        last_fixed_addr_sync_idx: list[int | None] = [None] * len(core_types)
        first_program_dispatched = False
        last_stall_idx: int | None = None
        subdevice_launch_window: list[int] = []

        for i, node in enumerate(trace_nodes):
            if node.sub_device_id != sub_device_id:
                continue

            node_metadata = metadata[i]
            nonbinary_sync_idx: int | None = None
            binary_sync_idx: int | None = None
            all_binaries_cached = True

            for core_info in core_types:
                index = core_info.index
                if core_info.skip:
                    continue

                program_config = node.per_core_type[index]
                allocator = region_allocators[index]
                program_id = node.program_id

                nonbinary_size = program_config.nonbinary_size
                binary_size = program_config.binary_size

                rta_sync_idx, rta_addr = allocator.allocate_region(
                    nonbinary_size, i, SIMPLE_NONBINARY, program_id
                )
                nonbinary_sync_idx = merge_syncs(nonbinary_sync_idx, rta_sync_idx)
                if nonbinary_size > 0:
                    node_metadata.dispatch_bytes += nonbinary_size

                binary_addr = 0
                if core_info.has_separate_binary_offset and core_info.binary_in_config and binary_size > 0:
                    cached_binary_addr = allocator.get_region(SIMPLE_BINARY, program_id)
                    if cached_binary_addr is not None:
                        binary_addr = cached_binary_addr
                        allocator.update_region_trace_idx(cached_binary_addr, i)
                    else:
                        all_binaries_cached = False
                        res_sync_idx, res_addr = allocator.allocate_region(binary_size, i, SIMPLE_BINARY, program_id)
                        if res_addr is None:
                            allocator.reset_allocator()
                            rta_sync_idx, rta_addr = allocator.allocate_region(
                                nonbinary_size, i, SIMPLE_NONBINARY, program_id
                            )
                            res_sync_idx, res_addr = allocator.allocate_region(
                                binary_size, i, SIMPLE_BINARY, program_id
                            )
                            if res_addr is None:
                                raise RuntimeError(f"failed to allocate binary region at trace index {i}")
                            if not subdevice_launch_window:
                                raise RuntimeError(f"failed to allocate binary region on first trace index {i}")
                            last_subdevice_idx = subdevice_launch_window[-1]
                            binary_sync_idx = merge_syncs(binary_sync_idx, last_subdevice_idx)
                            nonbinary_sync_idx = merge_syncs(nonbinary_sync_idx, last_subdevice_idx)
                        else:
                            binary_sync_idx = merge_syncs(binary_sync_idx, res_sync_idx)
                        binary_addr = res_addr
                        allocator.add_region(SIMPLE_BINARY, program_id, binary_addr)
                        node_metadata.dispatch_bytes += binary_size
                elif not core_info.binary_in_config and program_config.has_kernel_groups:
                    all_binaries_cached = False
                    if last_fixed_addr_sync_idx[index] is not None:
                        binary_sync_idx = merge_syncs(binary_sync_idx, last_fixed_addr_sync_idx[index])
                    last_fixed_addr_sync_idx[index] = i
                    node_metadata.dispatch_bytes += binary_size
                elif not core_info.has_separate_binary_offset and program_config.has_kernel_groups:
                    all_binaries_cached = False

                if rta_addr is None:
                    raise RuntimeError(f"failed to allocate non-binary region at trace index {i}")
                node_metadata.nonbinary_addrs[index] = rta_addr + ringbuffer_configs[index].start
                node_metadata.binary_addrs[index] = binary_addr + ringbuffer_configs[index].start

            node_metadata.send_binary = not all_binaries_cached
            extra_data[i].finished_sync_count = expected_workers_completed + node.num_workers

            max_queued_programs = LAUNCH_MSG_BUFFER_NUM_ENTRIES - 1
            if len(subdevice_launch_window) >= max_queued_programs:
                binary_sync_idx = merge_syncs(binary_sync_idx, subdevice_launch_window[0])

            if not first_program_dispatched:
                node_metadata.sync_count = 0
                node_metadata.stall_first = True
                first_program_dispatched = True

            needs_nonbinary_sync = nonbinary_sync_idx is not None and (
                last_stall_idx is None or nonbinary_sync_idx > last_stall_idx
            )
            needs_binary_sync = binary_sync_idx is not None and (
                last_stall_idx is None or binary_sync_idx > last_stall_idx
            )
            if needs_nonbinary_sync or needs_binary_sync:
                combined_sync_idx = merge_syncs(nonbinary_sync_idx, binary_sync_idx)
                assert combined_sync_idx is not None
                node_metadata.sync_count = extra_data[combined_sync_idx].finished_sync_count
                if needs_nonbinary_sync:
                    node_metadata.stall_first = True
                else:
                    node_metadata.stall_before_program = True
                node_metadata.stall_idx = combined_sync_idx
                last_stall_idx = combined_sync_idx

            expected_workers_completed += node.num_workers
            subdevice_launch_window.append(i)
            if len(subdevice_launch_window) > max_queued_programs:
                subdevice_launch_window.pop(0)

    return metadata


class TraceCacheWorkerBufferManager:
    def __init__(self, buffer_size: int, reuse_window: int) -> None:
        self.buffer_size = buffer_size
        self.reuse_window = reuse_window
        self.program_data_alloced: dict[int, bool] = {}
        self.allocator: list[TraceCacheAllocNode] = []
        self.lru: list[TraceCacheAllocNode] = []
        self.alloced_programs: dict[int, TraceCacheAllocNode | None] = {}

    def process_trace(
        self, trace: list[TraceCacheTraceNode], programs: dict[int, TraceCacheProgram]
    ) -> list[TraceCacheTraceNode]:
        self.program_data_alloced = {program_id: False for program_id in programs}
        self.allocator.clear()
        self.lru.clear()
        self.alloced_programs = {program_id: None for program_id in programs}
        self._build_use_data(trace, programs)
        if not self._handle_trivial_cases(trace, programs):
            self._alloc(trace, programs)
        return trace

    def _build_use_data(self, trace: list[TraceCacheTraceNode], programs: dict[int, TraceCacheProgram]) -> None:
        used_idx: dict[int, int] = {}
        counts: dict[int, int] = {}
        max_cost = max((program.cost for program in programs.values()), default=0)

        for trace_idx in range(len(trace) - 1, -1, -1):
            program_id = trace[trace_idx].program_id
            size_needed = programs[program_id].size
            if size_needed > self.buffer_size:
                raise RuntimeError(
                    f"program {program_id}'s size {size_needed} exceeds buffer size {self.buffer_size}"
                )
            if program_id in used_idx:
                trace[trace_idx].next_idx = used_idx[program_id]
            trace[trace_idx].remaining = counts.get(program_id, 0)
            used_idx[program_id] = trace_idx
            counts[program_id] = counts.get(program_id, 0) + 1

        for trace_node in trace:
            cost = programs[trace_node.program_id].cost
            normalized_cost = 0.0 if max_cost == 0 else cost / max_cost
            trace_node.weight = normalized_cost * trace_node.remaining

    def _handle_trivial_cases(
        self, trace: list[TraceCacheTraceNode], programs: dict[int, TraceCacheProgram]
    ) -> bool:
        total_alloced = sum(programs[trace_node.program_id].size for trace_node in trace)
        if total_alloced > self.buffer_size:
            return False

        alloc_addr = 0
        for trace_idx, trace_node in enumerate(trace):
            if trace_node.addr is not None:
                continue
            program_id = trace_node.program_id
            child_idx: int | None = trace_idx
            while child_idx is not None:
                trace[child_idx].addr = alloc_addr
                child_idx = trace[child_idx].next_idx
            alloc_addr += programs[program_id].size
            self.program_data_alloced[program_id] = True
            trace_node.does_dispatch = True
        return True

    def _alloc(self, trace: list[TraceCacheTraceNode], programs: dict[int, TraceCacheProgram]) -> None:
        trace_idx = 0
        eviction_mode = False
        pre_alloc_addr_top = self.buffer_size
        pre_alloc_addr = self.buffer_size
        uncommitted_marker: TraceCacheAllocNode | None = None

        while trace_idx < len(trace):
            program_id = trace[trace_idx].program_id
            alloc_node = self.alloced_programs.get(program_id)

            if alloc_node is not None:
                trace[trace_idx].addr = alloc_node.addr
                self._move_lru_to_back(alloc_node)
                alloc_node.prev_use = trace_idx
            else:
                size_needed = programs[program_id].size
                if eviction_mode:
                    evict_node = self._find_eviction_candidates(True, self.reuse_window, size_needed, trace_idx, trace)
                    if evict_node is None:
                        for window in range(self.reuse_window, 0, -1):
                            evict_node = self._find_eviction_candidates(False, window, size_needed, trace_idx, trace)
                            if evict_node is not None:
                                break
                    if evict_node is None:
                        raise RuntimeError(f"failed to allocate {size_needed} bytes at trace index {trace_idx}")

                    freed_size, alloc_index = self._evict(evict_node, trace_idx, size_needed, trace)
                    if not self.allocator:
                        pre_alloc_addr_top = self.buffer_size
                        pre_alloc_addr = self.buffer_size
                        uncommitted_marker = None
                        eviction_mode = False
                        continue
                    self._allocate_in_hole(trace_idx, freed_size, size_needed, alloc_index, trace)
                else:
                    if pre_alloc_addr < size_needed:
                        self._sort_preallocations(uncommitted_marker, trace)
                        self._commit_preallocations(pre_alloc_addr_top, uncommitted_marker, trace)
                        (
                            eviction_mode,
                            pre_alloc_addr_top,
                            pre_alloc_addr,
                            uncommitted_marker,
                        ) = self._try_to_reenter_preallocation_mode(
                            pre_alloc_addr_top, pre_alloc_addr, uncommitted_marker, trace_idx, trace
                        )
                        continue

                    pre_alloc_addr -= size_needed
                    new_node = TraceCacheAllocNode(program_id, pre_alloc_addr, size_needed, trace_idx, trace_idx)
                    self.allocator.append(new_node)
                    self.lru.append(new_node)
                    self.alloced_programs[program_id] = new_node

            trace_idx += 1

        if not eviction_mode:
            self._commit_preallocations(pre_alloc_addr_top, uncommitted_marker, trace)

    def _find_eviction_candidates(
        self,
        only_stale: bool,
        window: int,
        size_needed: int,
        trace_idx: int,
        trace: list[TraceCacheTraceNode],
    ) -> TraceCacheAllocNode | None:
        match: TraceCacheAllocNode | None = None
        best_score = math.inf

        for lru_node in self.lru:
            free_size = 0
            found_one = False
            score = 0.0
            alloc_index = self._allocator_index(lru_node)

            for alloc_node in self.allocator[alloc_index:]:
                if alloc_node.prev_use + window > trace_idx:
                    break

                weight = alloc_node.weight(trace)
                if not only_stale or weight == 0.0:
                    free_size += alloc_node.size
                    score += weight
                    if size_needed <= free_size:
                        found_one = True
                        break

            if found_one and score < best_score:
                best_score = score
                match = lru_node

        return match

    def _evict(
        self,
        evict_node: TraceCacheAllocNode,
        trace_idx: int,
        size_needed: int,
        trace: list[TraceCacheTraceNode],
    ) -> tuple[int, int]:
        freed_size = 0
        alloc_index = self._allocator_index(evict_node)

        while freed_size < size_needed:
            alloc_node = self.allocator[alloc_index]
            freed_size += alloc_node.size
            if not alloc_node.is_free:
                self.program_data_alloced[alloc_node.program_id] = False
                self.alloced_programs[alloc_node.program_id] = None

            if trace[trace_idx].stall_idx is None or alloc_node.prev_use > trace[trace_idx].stall_idx:
                trace[trace_idx].stall_idx = alloc_node.prev_use

            self._remove_lru(alloc_node)
            del self.allocator[alloc_index]

        return freed_size, alloc_index

    def _allocate_in_hole(
        self,
        trace_idx: int,
        freed_size: int,
        size_needed: int,
        alloc_index: int,
        trace: list[TraceCacheTraceNode],
    ) -> None:
        program_id = trace[trace_idx].program_id

        if alloc_index == len(self.allocator):
            prev_node = self.allocator[-1]
            alloc_addr = prev_node.addr - size_needed
            new_node = TraceCacheAllocNode(program_id, alloc_addr, size_needed, trace_idx, trace_idx)
            self.allocator.append(new_node)
            self.lru.append(new_node)
            if alloc_addr != 0:
                free_node = TraceCacheAllocNode.free(0, alloc_addr)
                self.allocator.append(free_node)
                self.lru.insert(0, free_node)
        else:
            below_node = self.allocator[alloc_index]
            base_addr = below_node.addr + below_node.size

            if freed_size == size_needed:
                alloc_addr = base_addr
                new_node = TraceCacheAllocNode(program_id, alloc_addr, size_needed, trace_idx, trace_idx)
                self.allocator.insert(alloc_index, new_node)
                self.lru.append(new_node)
            else:
                hole_top = False
                if alloc_index != 0:
                    above_node = self.allocator[alloc_index - 1]
                    hole_top = above_node.weight(trace) < below_node.weight(trace)

                hole_size = freed_size - size_needed
                if hole_top:
                    alloc_addr = base_addr
                    insert_index = alloc_index
                    above_node = self.allocator[alloc_index - 1]
                    if above_node.is_free:
                        above_node.size += hole_size
                        above_node.addr = base_addr + size_needed
                    else:
                        free_node = TraceCacheAllocNode.free(base_addr + size_needed, hole_size)
                        self.allocator.insert(insert_index, free_node)
                        self.lru.insert(0, free_node)
                        insert_index += 1
                    new_node = TraceCacheAllocNode(program_id, alloc_addr, size_needed, trace_idx, trace_idx)
                    self.allocator.insert(insert_index, new_node)
                    self.lru.append(new_node)
                else:
                    alloc_addr = base_addr + hole_size
                    new_node = TraceCacheAllocNode(program_id, alloc_addr, size_needed, trace_idx, trace_idx)
                    self.allocator.insert(alloc_index, new_node)
                    self.lru.append(new_node)
                    alloc_index += 1
                    below_node = self.allocator[alloc_index]
                    if below_node.is_free:
                        below_node.size += hole_size
                    else:
                        free_node = TraceCacheAllocNode.free(base_addr, hole_size)
                        self.allocator.insert(alloc_index, free_node)
                        self.lru.insert(0, free_node)

        trace[trace_idx].does_dispatch = True
        trace[trace_idx].addr = alloc_addr
        self.program_data_alloced[program_id] = True
        self.alloced_programs[program_id] = new_node

    def _sort_preallocations(self, uncommitted_marker: TraceCacheAllocNode | None, trace: list[TraceCacheTraceNode]) -> None:
        start = 0 if uncommitted_marker is None else self._allocator_index(uncommitted_marker) + 1
        sorted_tail = sorted(self.allocator[start:], key=lambda node: (-node.weight(trace), -node.prev_use))
        self.allocator[start:] = sorted_tail

    def _commit_preallocations(
        self, addr_top: int, commit_start: TraceCacheAllocNode | None, trace: list[TraceCacheTraceNode]
    ) -> None:
        addr = addr_top
        start = 0 if commit_start is None else self._allocator_index(commit_start) + 1
        for alloc_node in self.allocator[start:]:
            if alloc_node.is_free:
                continue

            trace_idx = alloc_node.first_use
            program_id = alloc_node.program_id
            addr -= alloc_node.size
            alloc_node.addr = addr
            self.program_data_alloced[program_id] = True
            trace[trace_idx].does_dispatch = True

            child_idx: int | None = trace_idx
            while child_idx is not None and child_idx <= alloc_node.prev_use:
                trace[child_idx].addr = alloc_node.addr
                child_idx = trace[child_idx].next_idx

    def _try_to_reenter_preallocation_mode(
        self,
        pre_alloc_addr_top: int,
        pre_alloc_addr: int,
        uncommitted_marker: TraceCacheAllocNode | None,
        trace_idx: int,
        trace: list[TraceCacheTraceNode],
    ) -> tuple[bool, int, int, TraceCacheAllocNode | None]:
        eviction_mode = True
        done = False
        stall_idx: int | None = None

        while not done and self.allocator:
            node = self.allocator[-1]
            last_use = node.prev_use
            pre_alloc_addr = node.addr
            pre_alloc_addr_top = node.addr

            if last_use + self.reuse_window >= trace_idx or trace[last_use].remaining != 0:
                done = True
            else:
                self.program_data_alloced[node.program_id] = False
                self.alloced_programs[node.program_id] = None
                stall_idx = node.prev_use
                self._remove_lru(node)
                self.allocator.pop()
                eviction_mode = False

        if eviction_mode:
            if self.allocator:
                addr = self.allocator[-1].addr
                if addr > 0:
                    free_node = TraceCacheAllocNode.free(0, addr)
                    self.allocator.append(free_node)
                    self.lru.insert(0, free_node)
        else:
            uncommitted_marker = self.allocator[-1] if self.allocator else None

        trace[trace_idx].stall_idx = stall_idx
        return eviction_mode, pre_alloc_addr_top, pre_alloc_addr, uncommitted_marker

    def _allocator_index(self, node: TraceCacheAllocNode) -> int:
        for index, candidate in enumerate(self.allocator):
            if candidate is node:
                return index
        raise RuntimeError("allocator node not found")

    def _remove_lru(self, node: TraceCacheAllocNode) -> None:
        for index, candidate in enumerate(self.lru):
            if candidate is node:
                del self.lru[index]
                return
        raise RuntimeError("LRU node not found")

    def _move_lru_to_back(self, node: TraceCacheAllocNode) -> None:
        self._remove_lru(node)
        self.lru.append(node)


def trace_cache_program_size(core_info: CoreTypeInfo, per_core: PerCoreTypeNode) -> int:
    if core_info.skip:
        return 0
    if core_info.has_separate_binary_offset and core_info.binary_in_config:
        return per_core.nonbinary_size + per_core.binary_size
    return per_core.nonbinary_size


def build_trace_cache_inputs(
    core_info: CoreTypeInfo, trace_nodes: list[TraceNodeInput]
) -> tuple[list[TraceCacheTraceNode], dict[int, TraceCacheProgram]]:
    trace = [TraceCacheTraceNode(node.program_id) for node in trace_nodes]
    programs: dict[int, TraceCacheProgram] = {}
    for node in trace_nodes:
        per_core = node.per_core_type[core_info.index]
        size = trace_cache_program_size(core_info, per_core)
        if size == 0:
            continue
        cost = size
        existing = programs.get(node.program_id)
        if existing is not None and existing.size != size:
            raise RuntimeError(
                f"program {node.program_id} has inconsistent size for core {core_info.index}: "
                f"{existing.size} vs {size}"
            )
        programs[node.program_id] = TraceCacheProgram(size=size, cost=cost)
    return trace, programs


def run_trace_cache2_allocator(
    core_types: list[CoreTypeInfo],
    ringbuffer_configs: list[RingbufferConfig],
    trace_nodes: list[TraceNodeInput],
    reuse_window: int,
) -> tuple[list[int | None], list[int], dict[int, list[TraceCacheTraceNode]]]:
    combined_stalls: list[int | None] = [None] * len(trace_nodes)
    dispatch_bytes: list[int] = [0] * len(trace_nodes)
    per_core_traces: dict[int, list[TraceCacheTraceNode]] = {}

    sub_device_ids = sorted({node.sub_device_id for node in trace_nodes})
    for core_info in core_types:
        if core_info.skip:
            continue

        per_core_full_trace, programs = build_trace_cache_inputs(core_info, trace_nodes)
        if not programs:
            continue

        for sub_device_id in sub_device_ids:
            subdevice_indices = [
                index for index, node in enumerate(trace_nodes) if node.sub_device_id == sub_device_id
            ]
            subdevice_trace = [copy.copy(per_core_full_trace[index]) for index in subdevice_indices]
            manager = TraceCacheWorkerBufferManager(ringbuffer_configs[core_info.index].size, reuse_window)
            manager.process_trace(subdevice_trace, programs)

            for local_index, global_index in enumerate(subdevice_indices):
                per_core_full_trace[global_index] = subdevice_trace[local_index]

        per_core_traces[core_info.index] = per_core_full_trace
        for trace_idx, trace_node in enumerate(per_core_full_trace):
            if trace_node.stall_idx is not None:
                combined_stalls[trace_idx] = merge_syncs(combined_stalls[trace_idx], trace_node.stall_idx)
            if trace_node.does_dispatch:
                dispatch_bytes[trace_idx] += programs[trace_node.program_id].size

    return combined_stalls, dispatch_bytes, per_core_traces


def compute_stats(name: str, stalls: list[int | None], dispatch_bytes: list[int]) -> AllocatorStats:
    badness = 0.0
    stall_count = 0
    immediate_previous_stalls = 0
    min_distance: int | None = None
    max_distance: int | None = None

    for trace_idx, stall_idx in enumerate(stalls):
        if stall_idx is None or stall_idx < 0:
            continue
        if stall_idx >= trace_idx:
            raise RuntimeError(f"{name} has invalid stall at trace {trace_idx} on {stall_idx}")

        stall_node_gap = trace_idx - stall_idx
        distance = stall_node_gap - 1
        if stall_node_gap >= LAUNCH_MSG_BUFFER_NUM_ENTRIES - 1:
            contribution = 0.0
        else:
            contribution = math.ldexp(1.0, -distance)
        badness += contribution
        stall_count += 1
        if distance == 0:
            immediate_previous_stalls += 1
        min_distance = distance if min_distance is None else min(min_distance, distance)
        max_distance = distance if max_distance is None else max(max_distance, distance)

    return AllocatorStats(
        name=name,
        nodes=len(stalls),
        stalls=stall_count,
        badness=badness,
        immediate_previous_stalls=immediate_previous_stalls,
        min_stall_distance=min_distance,
        max_stall_distance=max_distance,
        dispatch_bytes=sum(dispatch_bytes),
    )


def validate_simple_results(simple_metadata: list[SimpleMetadata], captured_results: list[dict[str, Any]]) -> list[str]:
    mismatches: list[str] = []
    if not captured_results:
        return mismatches

    fields = ("binary_addrs", "nonbinary_addrs", "send_binary", "stall_before_program", "stall_first", "sync_count")
    for trace_idx, (actual, expected) in enumerate(zip(simple_metadata, captured_results)):
        actual_dict = {
            "binary_addrs": actual.binary_addrs,
            "nonbinary_addrs": actual.nonbinary_addrs,
            "send_binary": actual.send_binary,
            "stall_before_program": actual.stall_before_program,
            "stall_first": actual.stall_first,
            "sync_count": actual.sync_count,
        }
        for field_name in fields:
            if actual_dict[field_name] != expected[field_name]:
                mismatches.append(
                    f"trace {trace_idx} field {field_name}: replay={actual_dict[field_name]!r}, "
                    f"captured={expected[field_name]!r}"
                )
                break
        if len(mismatches) >= 10:
            break

    if len(simple_metadata) != len(captured_results):
        mismatches.append(f"result length mismatch: replay={len(simple_metadata)}, captured={len(captured_results)}")

    return mismatches


def print_stats(stats: list[AllocatorStats]) -> None:
    print()
    print("Allocator comparison:")
    print(
        f"{'allocator':<28} {'stalls':>8} {'badness':>14} {'dist0':>8} "
        f"{'min_dist':>9} {'max_dist':>9} {'dispatch_bytes':>16}"
    )
    for item in stats:
        min_dist = "-" if item.min_stall_distance is None else str(item.min_stall_distance)
        max_dist = "-" if item.max_stall_distance is None else str(item.max_stall_distance)
        print(
            f"{item.name:<28} {item.stalls:>8} {item.badness:>14.6f} "
            f"{item.immediate_previous_stalls:>8} {min_dist:>9} {max_dist:>9} {item.dispatch_bytes:>16}"
        )


def print_worst_stalls(name: str, stalls: list[int | None], limit: int) -> None:
    if limit <= 0:
        return
    ranked: list[tuple[float, int, int, int]] = []
    for trace_idx, stall_idx in enumerate(stalls):
        if stall_idx is None or stall_idx < 0:
            continue
        stall_node_gap = trace_idx - stall_idx
        distance = stall_node_gap - 1
        if stall_node_gap >= LAUNCH_MSG_BUFFER_NUM_ENTRIES - 1:
            contribution = 0.0
        else:
            contribution = math.ldexp(1.0, -distance)
        ranked.append((contribution, distance, trace_idx, stall_idx))

    ranked.sort(reverse=True)
    print()
    print(f"Worst {name} stalls:")
    for contribution, distance, trace_idx, stall_idx in ranked[:limit]:
        print(
            f"  trace {trace_idx} stalls on {stall_idx}: "
            f"stall_distance={distance}, contribution={contribution:.6f}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "trace_json",
        nargs="?",
        default="/home/jbauman/trace_alloc_9.json",
        type=Path,
        help="Trace allocation capture JSON (default: /home/jbauman/trace_alloc_9.json).",
    )
    parser.add_argument(
        "--trace-cache-reuse-window",
        type=int,
        default=TRACE_CACHE_MAX_REUSE_WINDOW,
        help=f"trace-cache2 reuse window to replay (default: {TRACE_CACHE_MAX_REUSE_WINDOW}).",
    )
    parser.add_argument(
        "--dump-worst-stalls",
        type=int,
        default=0,
        help="Print this many highest-contribution stalls for each allocator.",
    )
    args = parser.parse_args()

    core_types, ringbuffer_configs, trace_nodes, captured_results = load_capture(args.trace_json)
    print(
        f"Loaded {args.trace_json}: {len(trace_nodes)} trace nodes, "
        f"{len({node.program_id for node in trace_nodes})} programs, {len(core_types)} core types."
    )

    simple_metadata = run_simple_allocator(core_types, ringbuffer_configs, trace_nodes)
    simple_mismatches = validate_simple_results(simple_metadata, captured_results)
    if simple_mismatches:
        print("SimpleTraceAllocator replay does not match captured results:", file=sys.stderr)
        for mismatch in simple_mismatches:
            print(f"  {mismatch}", file=sys.stderr)
        return 1
    if captured_results:
        print("SimpleTraceAllocator replay matches captured results.")

    simple_stalls = [metadata.stall_idx for metadata in simple_metadata]
    simple_dispatch_bytes = [metadata.dispatch_bytes for metadata in simple_metadata]
    trace_cache_stalls, trace_cache_dispatch_bytes, per_core_trace_cache = run_trace_cache2_allocator(
        core_types, ringbuffer_configs, trace_nodes, args.trace_cache_reuse_window
    )

    stats = [
        compute_stats("SimpleTraceAllocator", simple_stalls, simple_dispatch_bytes),
        compute_stats("trace-cache2", trace_cache_stalls, trace_cache_dispatch_bytes),
    ]
    print_stats(stats)

    per_core_stats: list[AllocatorStats] = []
    for core_info in core_types:
        trace = per_core_trace_cache.get(core_info.index)
        if trace is None:
            continue
        _, programs = build_trace_cache_inputs(core_info, trace_nodes)
        dispatch_bytes = [
            programs[node.program_id].size if node.does_dispatch else 0
            for node in trace
        ]
        per_core_stats.append(
            compute_stats(
                f"trace-cache2 core {core_info.index} {core_info.core_type}",
                [node.stall_idx for node in trace],
                dispatch_bytes,
            )
        )
    if per_core_stats:
        print_stats(per_core_stats)

    print_worst_stalls("SimpleTraceAllocator", simple_stalls, args.dump_worst_stalls)
    print_worst_stalls("trace-cache2", trace_cache_stalls, args.dump_worst_stalls)

    return 0


if __name__ == "__main__":
    sys.exit(main())
