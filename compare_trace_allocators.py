#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Standalone reimplementation of SimpleTraceAllocator and LegacyTraceAllocator
(the WorkerConfigBufferMgr-based allocator) to compare their allocation
decisions on the same trace_alloc JSON input.

No HAL or C++ dependencies required -- all logic is in pure Python.
"""

import json
import sys
from dataclasses import dataclass, field
from typing import Optional
import math


LAUNCH_MSG_BUFFER_NUM_ENTRIES = 8
KERNEL_CONFIG_ENTRY_COUNT = 8
ACTIVE_ETH_IDX = 1  # HalProgrammableCoreType::ACTIVE_ETH


# ─────────────────────────────────────────────────────────────
# Data model mirroring the JSON schema
# ─────────────────────────────────────────────────────────────
@dataclass
class CoreTypeInfo:
    index: int
    core_type: str
    binary_in_config: bool
    has_separate_binary_offset: bool
    skip: bool


@dataclass
class PerCoreTypeInput:
    index: int
    nonbinary_size: int
    binary_size: int
    has_kernel_groups: bool


@dataclass
class TraceNodeInput:
    program_id: int
    sub_device_id: int
    num_workers: int
    per_core_type: list  # list[PerCoreTypeInput]


@dataclass
class AllocResult:
    nonbinary_addrs: list  # list[int] per core type
    binary_addrs: list  # list[int] per core type
    send_binary: bool
    sync_count: int
    stall_first: bool
    stall_before_program: bool


# ─────────────────────────────────────────────────────────────
# WorkerConfigBufferMgr reimplementation (used by Legacy)
# ─────────────────────────────────────────────────────────────
class WorkerConfigBufferMgr:
    def __init__(self):
        self.base_addrs = []
        self.end_addrs = []
        self.entries = [[None] * 0 for _ in range(KERNEL_CONFIG_ENTRY_COUNT)]
        self.alloc_index = []
        self.free_index = []
        self.reservation = []

    def init_add_buffer(self, base_addr: int, size: int):
        idx = len(self.base_addrs)
        self.base_addrs.append(base_addr)
        self.end_addrs.append(base_addr + size)
        for entry_list in self.entries:
            entry_list.append({"addr": 0, "size": 0, "sync_count": 0})
        self.alloc_index.append(0)
        self.free_index.append(0)
        self.entries[0][idx] = {"addr": base_addr, "size": 0, "sync_count": 0}
        self.reservation.append({"addr": 0, "size": 0})

    def reserve(self, sizes: list):
        need_sync = False
        sync_count = 0
        num_buffer_types = len(self.reservation)
        assert len(sizes) == num_buffer_types

        for idx in range(num_buffer_types):
            free_index = self.free_index[idx]
            alloc_index = self.alloc_index[idx]

            if idx == ACTIVE_ETH_IDX and sizes[idx]:
                if free_index != alloc_index:
                    need_sync = True
                    sync_count = max(sync_count, self.entries[free_index][idx]["sync_count"])
                self.reservation[idx]["addr"] = self.base_addrs[idx]
                self.reservation[idx]["size"] = sizes[idx]
                continue

            done = False
            while not done:
                done = True
                size = sizes[idx]
                addr = self.entries[alloc_index][idx]["addr"]
                self.reservation[idx]["size"] = size

                if size == 0:
                    self.reservation[idx]["addr"] = addr
                    break

                if free_index == alloc_index:
                    self.reservation[idx]["addr"] = self.base_addrs[idx]
                    break

                assert size <= self.end_addrs[idx] - self.base_addrs[idx]

                free_entry = self.entries[free_index][idx]
                if addr >= free_entry["addr"] + free_entry["size"]:
                    end = self.end_addrs[idx]
                else:
                    end = free_entry["addr"]

                if addr + size > end and end == self.end_addrs[idx]:
                    addr = self.base_addrs[idx]
                    end = free_entry["addr"]

                had_sync = need_sync
                if addr + size > end:
                    next_free_index = (free_index + 1) % KERNEL_CONFIG_ENTRY_COUNT

                    if next_free_index == alloc_index:
                        addr = self.base_addrs[idx]
                    else:
                        next_entry = self.entries[next_free_index][idx]
                        if addr >= next_entry["addr"]:
                            next_end = self.end_addrs[idx]
                        else:
                            next_end = next_entry["addr"]
                        if addr + size > next_end:
                            free_index = next_free_index
                            done = False
                            continue

                    need_sync = True
                    sc = self.entries[free_index][idx]["sync_count"]
                    sync_count = max(sync_count, sc) if had_sync else sc
                elif (alloc_index + 1 == free_index or
                      (alloc_index + 1 == KERNEL_CONFIG_ENTRY_COUNT and free_index == 0)):
                    need_sync = True
                    sc = self.entries[free_index][idx]["sync_count"]
                    sync_count = max(sync_count, sc) if had_sync else sc

                self.reservation[idx]["addr"] = addr

        return (need_sync, sync_count), self.reservation

    def free(self, free_up_to_sync_count: int):
        for idx in range(len(self.reservation)):
            fi = self.free_index[idx]
            while (free_up_to_sync_count >= self.entries[fi][idx]["sync_count"] and
                   fi != self.alloc_index[idx]):
                fi = (fi + 1) % KERNEL_CONFIG_ENTRY_COUNT
                self.free_index[idx] = fi

    def alloc(self, when_freeable_sync_count: int):
        for idx in range(len(self.reservation)):
            if self.reservation[idx]["size"] == 0:
                continue
            ai = self.alloc_index[idx]
            self.entries[ai][idx]["addr"] = self.reservation[idx]["addr"]
            self.entries[ai][idx]["size"] = self.reservation[idx]["size"]
            self.entries[ai][idx]["sync_count"] = when_freeable_sync_count
            old_ai = ai
            ai = (ai + 1) % KERNEL_CONFIG_ENTRY_COUNT
            self.entries[ai][idx]["addr"] = (
                self.entries[old_ai][idx]["addr"] + self.entries[old_ai][idx]["size"])
            self.entries[ai][idx]["size"] = 0
            self.entries[ai][idx]["sync_count"] = 0xbabababa
            self.alloc_index[idx] = ai

    def mark_completely_full(self, sync: int):
        for idx in range(len(self.reservation)):
            new_free = 0
            new_alloc = 1
            self.alloc_index[idx] = new_alloc
            self.free_index[idx] = new_free
            self.entries[new_free][idx]["addr"] = self.base_addrs[idx]
            self.entries[new_free][idx]["size"] = self.end_addrs[idx] - self.base_addrs[idx]
            self.entries[new_free][idx]["sync_count"] = sync
            self.entries[new_alloc][idx]["addr"] = self.end_addrs[idx]
            self.entries[new_alloc][idx]["size"] = 0
            self.entries[new_alloc][idx]["sync_count"] = 0xbabababa


# ─────────────────────────────────────────────────────────────
# Legacy trace allocator (WorkerConfigBufferMgr-based)
# ─────────────────────────────────────────────────────────────
def legacy_allocate(ringbuffer_configs, core_types, trace_nodes):
    programmable_core_count = len(core_types)
    results = []

    config_buffer_mgr = WorkerConfigBufferMgr()
    for cfg in ringbuffer_configs:
        config_buffer_mgr.init_add_buffer(cfg["start"], cfg["size"])
    config_buffer_mgr.init_add_buffer(0, LAUNCH_MSG_BUFFER_NUM_ENTRIES - 1)
    config_buffer_mgr.init_add_buffer(0, 1)

    expected_num_workers_completed = 0
    first_program_dispatched = False

    for node_idx, node in enumerate(trace_nodes):
        num_workers = node.num_workers
        previous_expected = expected_num_workers_completed
        wrapped = False
        if expected_num_workers_completed > 0xFFFFFFFF - num_workers:
            expected_num_workers_completed = 0
            previous_expected = 0
            wrapped = True
        expected_num_workers_completed += num_workers

        reset_worker_counts = False
        if wrapped:
            config_buffer_mgr.mark_completely_full(0)
            reset_worker_counts = True

        program_config_sizes = []
        runs_on_multicast = False
        runs_on_unicast = False
        for ct in core_types:
            pct = node.per_core_type[ct.index]
            program_config_sizes.append(pct.nonbinary_size + pct.binary_size)
            if pct.has_kernel_groups:
                if ct.core_type == "TENSIX":
                    runs_on_multicast = True
                elif ct.core_type == "ACTIVE_ETH":
                    runs_on_unicast = True
        program_config_sizes.append(1 if runs_on_multicast else 0)
        program_config_sizes.append(1 if runs_on_unicast else 0)

        (need_sync, sync_count_val), reservation = config_buffer_mgr.reserve(program_config_sizes)

        stall_first = need_sync
        stall_before_program = False

        if need_sync:
            config_sizes_no_launch = list(program_config_sizes)
            config_sizes_no_launch[-2] = 0
            config_sizes_no_launch[-1] = 0
            (mem_need_sync, _), _ = config_buffer_mgr.reserve(config_sizes_no_launch)
            if not mem_need_sync:
                stall_first = False
                stall_before_program = True
            (need_sync, sync_count_val), reservation = config_buffer_mgr.reserve(program_config_sizes)

        if stall_first or stall_before_program:
            config_buffer_mgr.free(sync_count_val)
        config_buffer_mgr.alloc(expected_num_workers_completed)

        nonbinary_addrs = []
        binary_addrs = []
        for ct in core_types:
            addr = reservation[ct.index]["addr"]
            nonbinary_addrs.append(addr)
            if ct.has_separate_binary_offset and ct.binary_in_config:
                pct = node.per_core_type[ct.index]
                binary_addrs.append(addr + pct.nonbinary_size)
            else:
                binary_addrs.append(addr)

        r = AllocResult(
            nonbinary_addrs=nonbinary_addrs,
            binary_addrs=binary_addrs,
            send_binary=True,
            sync_count=sync_count_val if need_sync else 0,
            stall_first=stall_first,
            stall_before_program=stall_before_program,
        )

        if not first_program_dispatched:
            r.sync_count = 0
            r.stall_first = True
            first_program_dispatched = True

        results.append(r)

    return results


# ─────────────────────────────────────────────────────────────
# SimpleTraceAllocator reimplementation
# ─────────────────────────────────────────────────────────────
K_NON_BINARY = 0
K_BINARY = 1
K_NUM_TYPES = 2


class RegionAllocator:
    def __init__(self, ringbuffer_size, extra_data):
        self.ringbuffer_size = ringbuffer_size
        self.extra_data = extra_data
        self.regions = {}  # addr -> MemoryUsage
        self.program_ids_memory_map = [{} for _ in range(K_NUM_TYPES)]  # data_type -> {pid -> addr}

    def reset(self):
        self.regions.clear()
        for m in self.program_ids_memory_map:
            m.clear()

    @staticmethod
    def intersects(begin_1, size_1, begin_2, size_2):
        return (begin_1 < begin_2 + size_2) and (begin_2 < begin_1 + size_1)

    def allocate_region(self, size, trace_idx, data_type, program_id):
        if size == 0:
            return None, 0

        best_addr = None
        best_cost = float("inf")
        best_region_sync_idx = None

        max_stall_history_size = LAUNCH_MSG_BUFFER_NUM_ENTRIES

        sorted_regions = sorted(self.regions.items())
        marked_for_deletion = set()

        positions = [0]
        for raddr, rusage in sorted_regions:
            pos = raddr + rusage["size"]
            if pos not in positions:
                positions.append(pos)
        positions.sort()

        for addr in positions:
            if addr + size > self.ringbuffer_size:
                break

            cost = 0.0
            region_sync_idx = None
            now_in_use = False

            for raddr, rusage in sorted_regions:
                if raddr >= addr + size:
                    break
                if self.intersects(addr, size, raddr, rusage["size"]):
                    if rusage["trace_idx"] == trace_idx:
                        now_in_use = True
                        break
                    next_use_idx = self.extra_data[rusage["trace_idx"]]["next_use_idx"][rusage["data_type"]]
                    if next_use_idx is not None:
                        if next_use_idx == trace_idx:
                            cost += 1000000000
                        else:
                            cost += rusage["size"] * 1.0 / (next_use_idx - trace_idx)
                    elif trace_idx - rusage["trace_idx"] > max_stall_history_size:
                        marked_for_deletion.add(raddr)
                    region_sync_idx = merge_syncs(region_sync_idx, rusage["trace_idx"])

            if not now_in_use:
                if region_sync_idx is not None:
                    desired_write_ahead = min(LAUNCH_MSG_BUFFER_NUM_ENTRIES, 7)
                    stall_badness = 100000000
                    region_idx_diff = trace_idx - region_sync_idx
                    if region_idx_diff < desired_write_ahead:
                        cost += stall_badness * (1 << (desired_write_ahead - region_idx_diff))

                if cost < best_cost:
                    best_cost = cost
                    best_addr = addr
                    best_region_sync_idx = region_sync_idx

                if cost == 0:
                    break

        for addr in marked_for_deletion:
            if addr in self.regions:
                usage = self.regions[addr]
                pid_map = self.program_ids_memory_map[usage["data_type"]]
                pid_map.pop(usage["program_id"], None)
                del self.regions[addr]

        if best_addr is None:
            return None, None

        to_erase = []
        for raddr, rusage in list(self.regions.items()):
            if self.intersects(best_addr, size, raddr, rusage["size"]):
                self.program_ids_memory_map[rusage["data_type"]].pop(rusage["program_id"], None)
                to_erase.append(raddr)
        for a in to_erase:
            del self.regions[a]

        self.regions[best_addr] = {
            "trace_idx": trace_idx,
            "data_type": data_type,
            "size": size,
            "program_id": program_id,
        }
        return best_region_sync_idx, best_addr

    def get_region(self, data_type, program_id):
        return self.program_ids_memory_map[data_type].get(program_id)

    def add_region(self, data_type, program_id, addr):
        self.program_ids_memory_map[data_type][program_id] = addr

    def update_region_trace_idx(self, region_addr, trace_idx):
        if region_addr in self.regions:
            self.regions[region_addr]["trace_idx"] = trace_idx


def merge_syncs(s1, s2):
    if s1 is not None and s2 is not None:
        return max(s1, s2)
    elif s1 is not None:
        return s1
    else:
        return s2


def simple_allocate(ringbuffer_configs, core_types, trace_nodes):
    programmable_core_count = len(core_types)
    n = len(trace_nodes)
    extra_data = [{"next_use_idx": [None, None], "finished_sync_count": 0} for _ in range(n)]

    allocators = []
    ringbuffer_starts = []
    for cfg in ringbuffer_configs:
        allocators.append(RegionAllocator(cfg["size"], extra_data))
        ringbuffer_starts.append(cfg["start"])

    program_ids_use_map = {}
    for i in range(n - 1, -1, -1):
        node = trace_nodes[i]
        pid = node.program_id
        if pid in program_ids_use_map:
            extra_data[i]["next_use_idx"][K_BINARY] = program_ids_use_map[pid]
        program_ids_use_map[pid] = i

    for alloc in allocators:
        alloc.reset()

    results = []
    expected_workers_completed = 0
    last_fixed_addr_sync_idx = [None] * programmable_core_count
    first_program_dispatched = False
    last_stall_idx = None
    subdevice_launch_window = []
    max_queued_programs = LAUNCH_MSG_BUFFER_NUM_ENTRIES - 1

    for i in range(n):
        node = trace_nodes[i]
        nonbinary_sync_idx = None
        binary_sync_idx = None
        all_binaries_cached = True

        nonbinary_addrs = [0] * programmable_core_count
        binary_addrs = [0] * programmable_core_count

        for ct in core_types:
            if ct.skip:
                continue
            pct = node.per_core_type[ct.index]
            has_sep = ct.has_separate_binary_offset
            bin_in_cfg = ct.binary_in_config

            if has_sep and bin_in_cfg:
                non_binary_size = pct.nonbinary_size
            else:
                non_binary_size = pct.nonbinary_size + pct.binary_size

            binary_size = pct.binary_size
            allocator = allocators[ct.index]
            pid = node.program_id

            rta_sync_idx, rta_addr = allocator.allocate_region(non_binary_size, i, K_NON_BINARY, pid)
            nonbinary_sync_idx = merge_syncs(nonbinary_sync_idx, rta_sync_idx)

            binary_addr = 0

            if has_sep and bin_in_cfg and binary_size > 0:
                cached = allocator.get_region(K_BINARY, pid)
                if cached is not None:
                    binary_addr = cached
                    allocator.update_region_trace_idx(cached, i)
                else:
                    all_binaries_cached = False
                    res_sync, res_addr = allocator.allocate_region(binary_size, i, K_BINARY, pid)
                    if res_addr is None:
                        allocator.reset()
                        rta_sync_idx, rta_addr = allocator.allocate_region(non_binary_size, i, K_NON_BINARY, pid)
                        res_sync, res_addr = allocator.allocate_region(binary_size, i, K_BINARY, pid)
                        assert res_addr is not None
                        assert len(subdevice_launch_window) > 0
                        last_sub_idx = subdevice_launch_window[-1]
                        binary_sync_idx = merge_syncs(binary_sync_idx, last_sub_idx)
                        nonbinary_sync_idx = merge_syncs(nonbinary_sync_idx, last_sub_idx)
                    else:
                        binary_sync_idx = merge_syncs(res_sync, binary_sync_idx)
                    binary_addr = res_addr
                    allocator.add_region(K_BINARY, pid, binary_addr)
            elif not bin_in_cfg and pct.has_kernel_groups:
                all_binaries_cached = False
                if last_fixed_addr_sync_idx[ct.index] is not None:
                    binary_sync_idx = merge_syncs(binary_sync_idx, last_fixed_addr_sync_idx[ct.index])
                last_fixed_addr_sync_idx[ct.index] = i
            elif not has_sep and pct.has_kernel_groups:
                all_binaries_cached = False

            assert rta_addr is not None
            nonbinary_addrs[ct.index] = rta_addr + ringbuffer_starts[ct.index]
            binary_addrs[ct.index] = binary_addr + ringbuffer_starts[ct.index]

        send_binary = not all_binaries_cached
        extra_data[i]["finished_sync_count"] = expected_workers_completed + node.num_workers

        if len(subdevice_launch_window) >= max_queued_programs:
            binary_sync_idx = merge_syncs(binary_sync_idx, subdevice_launch_window[0])

        r_sync_count = 0
        r_stall_first = False
        r_stall_before_program = False

        if not first_program_dispatched:
            r_sync_count = 0
            r_stall_first = True
            first_program_dispatched = True

        needs_nonbinary_sync = (nonbinary_sync_idx is not None and
                                (last_stall_idx is None or nonbinary_sync_idx > last_stall_idx))
        needs_binary_sync = (binary_sync_idx is not None and
                             (last_stall_idx is None or binary_sync_idx > last_stall_idx))

        if needs_nonbinary_sync or needs_binary_sync:
            combined_sync_idx = merge_syncs(nonbinary_sync_idx, binary_sync_idx)
            r_sync_count = extra_data[combined_sync_idx]["finished_sync_count"]
            if needs_nonbinary_sync:
                r_stall_first = True
            else:
                r_stall_before_program = True
            last_stall_idx = combined_sync_idx

        expected_workers_completed += node.num_workers
        subdevice_launch_window.append(i)
        if len(subdevice_launch_window) > max_queued_programs:
            subdevice_launch_window.pop(0)

        results.append(AllocResult(
            nonbinary_addrs=nonbinary_addrs,
            binary_addrs=binary_addrs,
            send_binary=send_binary,
            sync_count=r_sync_count,
            stall_first=r_stall_first,
            stall_before_program=r_stall_before_program,
        ))

    return results


# ─────────────────────────────────────────────────────────────
# Analysis helpers
# ─────────────────────────────────────────────────────────────
def count_stalls(results, label):
    stall_first_count = sum(1 for r in results if r.stall_first)
    stall_before_count = sum(1 for r in results if r.stall_before_program)
    send_binary_count = sum(1 for r in results if r.send_binary)

    # Classify stalls by how far back we're waiting.
    # A "short stall" is one where sync_count is close to the current cumulative
    # worker count -- i.e., we're waiting for a very recent program.
    cum_workers = 0
    short_stalls = []
    for i, r in enumerate(results):
        if r.stall_first or r.stall_before_program:
            lag = cum_workers - r.sync_count
            if lag < 72 * 7:  # within ~7 programs worth of workers (72 workers/program typical)
                short_stalls.append((i, r.sync_count, cum_workers, lag,
                                     "stall_first" if r.stall_first else "stall_before_program"))
        cum_workers += 72  # approximate, doesn't matter for classification

    print(f"\n{'='*70}")
    print(f" {label}")
    print(f"{'='*70}")
    print(f"  Total programs:          {len(results)}")
    print(f"  stall_first:             {stall_first_count}")
    print(f"  stall_before_program:    {stall_before_count}")
    print(f"  send_binary=True:        {send_binary_count}")
    print(f"  send_binary=False:       {len(results) - send_binary_count}")
    print(f"  Short stalls (< 7 progs lag): {len(short_stalls)}")

    return short_stalls


def detailed_stall_analysis(results, trace_nodes, label):
    """
    Classify stalls more precisely by measuring the gap between the sync_count
    and the cumulative workers completed at dispatch time.
    """
    cum_workers = 0
    stall_events = []
    for i, r in enumerate(results):
        if r.stall_first or r.stall_before_program:
            programs_back = 0
            w = cum_workers
            j = i - 1
            while j >= 0 and w > r.sync_count:
                w -= trace_nodes[j].num_workers
                programs_back += 1
                j -= 1
            stall_events.append({
                "index": i,
                "type": "stall_first" if r.stall_first else "stall_before_program",
                "sync_count": r.sync_count,
                "cum_workers": cum_workers,
                "programs_back": programs_back,
                "worker_lag": cum_workers - r.sync_count,
            })
        cum_workers += trace_nodes[i].num_workers

    if not stall_events:
        return stall_events

    buckets = {
        "0 (current)": [],
        "1-3": [],
        "4-7": [],
        "8+": [],
    }
    for ev in stall_events:
        pb = ev["programs_back"]
        if pb == 0:
            buckets["0 (current)"].append(ev)
        elif pb <= 3:
            buckets["1-3"].append(ev)
        elif pb <= 7:
            buckets["4-7"].append(ev)
        else:
            buckets["8+"].append(ev)

    print(f"\n  Stall depth distribution ({label}):")
    for bucket_name, evs in buckets.items():
        print(f"    {bucket_name:>15s} programs back: {len(evs):>5d}")
        if evs and len(evs) <= 10:
            for ev in evs:
                print(f"      node {ev['index']:>5d}: sync_count={ev['sync_count']}, "
                      f"cum_workers={ev['cum_workers']}, lag={ev['worker_lag']}")

    return stall_events


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "out/trace_alloc_9.json"
    with open(path) as f:
        data = json.load(f)

    core_types = [CoreTypeInfo(**ct) for ct in data["core_types"]]
    ringbuffer_configs = data["ringbuffer_configs"]
    trace_nodes = []
    for nd in data["trace_nodes"]:
        per_ct = [PerCoreTypeInput(**p) for p in nd["per_core_type"]]
        trace_nodes.append(TraceNodeInput(
            program_id=nd["program_id"],
            sub_device_id=nd["sub_device_id"],
            num_workers=nd["num_workers"],
            per_core_type=per_ct,
        ))

    # Validate our SimpleTraceAllocator reimplementation against the reference results.
    reference_results = data.get("results")
    simple_results = simple_allocate(ringbuffer_configs, core_types, trace_nodes)

    if reference_results:
        mismatches = 0
        for i, (ref, sim) in enumerate(zip(reference_results, simple_results)):
            match = True
            if ref["nonbinary_addrs"] != sim.nonbinary_addrs:
                match = False
            if ref["binary_addrs"] != sim.binary_addrs:
                match = False
            if ref["send_binary"] != sim.send_binary:
                match = False
            if ref["sync_count"] != sim.sync_count:
                match = False
            if ref["stall_first"] != sim.stall_first:
                match = False
            if ref["stall_before_program"] != sim.stall_before_program:
                match = False
            if not match:
                mismatches += 1
                if mismatches <= 10:
                    print(f"MISMATCH at node {i}:")
                    print(f"  Reference: {ref}")
                    print(f"  Simple:    nonbinary={sim.nonbinary_addrs} binary={sim.binary_addrs} "
                          f"send_binary={sim.send_binary} sync={sim.sync_count} "
                          f"stall_first={sim.stall_first} stall_before={sim.stall_before_program}")
        if mismatches == 0:
            print(f"SimpleTraceAllocator reimplementation matches reference for all {len(reference_results)} nodes.")
        else:
            print(f"\n{mismatches} mismatches out of {len(reference_results)} nodes.")

    # Run legacy allocator
    legacy_results = legacy_allocate(ringbuffer_configs, core_types, trace_nodes)

    # Analysis
    simple_short = count_stalls(simple_results, "SimpleTraceAllocator")
    legacy_short = count_stalls(legacy_results, "LegacyTraceAllocator")

    simple_events = detailed_stall_analysis(simple_results, trace_nodes, "SimpleTraceAllocator")
    legacy_events = detailed_stall_analysis(legacy_results, trace_nodes, "LegacyTraceAllocator")

    # Direct comparison of short stalls
    print(f"\n{'='*70}")
    print(f" HEAD-TO-HEAD COMPARISON")
    print(f"{'='*70}")

    # Compare stall points
    simple_stall_set = {i for i, r in enumerate(simple_results) if r.stall_first or r.stall_before_program}
    legacy_stall_set = {i for i, r in enumerate(legacy_results) if r.stall_first or r.stall_before_program}

    only_simple = simple_stall_set - legacy_stall_set
    only_legacy = legacy_stall_set - simple_stall_set
    both = simple_stall_set & legacy_stall_set

    print(f"  Stalls in both:             {len(both)}")
    print(f"  Stalls only in Simple:      {len(only_simple)}")
    print(f"  Stalls only in Legacy:      {len(only_legacy)}")

    # Binary sends comparison
    simple_binary_sends = sum(1 for r in simple_results if r.send_binary)
    legacy_binary_sends = sum(1 for r in legacy_results if r.send_binary)
    print(f"\n  Binary sends (Simple):      {simple_binary_sends}")
    print(f"  Binary sends (Legacy):      {legacy_binary_sends}")
    print(f"  Binary sends saved:         {legacy_binary_sends - simple_binary_sends}")

    # Show first few programs where they differ
    print(f"\n  First 20 differences in stall behavior:")
    diffs_shown = 0
    for i in range(len(trace_nodes)):
        sr = simple_results[i]
        lr = legacy_results[i]
        if (sr.stall_first != lr.stall_first or sr.stall_before_program != lr.stall_before_program or
                sr.sync_count != lr.sync_count or sr.send_binary != lr.send_binary):
            if diffs_shown < 20:
                print(f"    node {i:>5d} pid={trace_nodes[i].program_id}: "
                      f"Simple(sf={sr.stall_first}, sb={sr.stall_before_program}, sync={sr.sync_count}, "
                      f"bin={sr.send_binary}) vs "
                      f"Legacy(sf={lr.stall_first}, sb={lr.stall_before_program}, sync={lr.sync_count}, "
                      f"bin={lr.send_binary})")
                diffs_shown += 1
    if diffs_shown == 0:
        print("    (none)")

    # Detailed: show the short stalls (depth < 7 programs) for both
    print(f"\n{'='*70}")
    print(f" SHORT STALLS DETAIL (stall waits for a program <=7 dispatches ago)")
    print(f"{'='*70}")
    simple_short_events = [e for e in simple_events if e["programs_back"] <= 7]
    legacy_short_events = [e for e in legacy_events if e["programs_back"] <= 7]

    print(f"\n  SimpleTraceAllocator short stalls: {len(simple_short_events)}")
    for ev in simple_short_events[:30]:
        print(f"    node {ev['index']:>5d}: {ev['type']:25s} sync_count={ev['sync_count']:>7d} "
              f"programs_back={ev['programs_back']} worker_lag={ev['worker_lag']}")
    if len(simple_short_events) > 30:
        print(f"    ... and {len(simple_short_events) - 30} more")

    print(f"\n  LegacyTraceAllocator short stalls: {len(legacy_short_events)}")
    for ev in legacy_short_events[:30]:
        print(f"    node {ev['index']:>5d}: {ev['type']:25s} sync_count={ev['sync_count']:>7d} "
              f"programs_back={ev['programs_back']} worker_lag={ev['worker_lag']}")
    if len(legacy_short_events) > 30:
        print(f"    ... and {len(legacy_short_events) - 30} more")


if __name__ == "__main__":
    main()
