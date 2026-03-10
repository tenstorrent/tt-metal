# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
General utilities for the GPT-OSS demo.
"""

import ttnn


def get_cache_file_name(tensor_cache_path, name):
    return f"{tensor_cache_path}/{name}" if tensor_cache_path else None


saved_address = []


def print_memory_usage(device, save_address=False, diff_address=False):
    device_info = ttnn._ttnn.reports.get_buffers(device)
    dram_usage = 0
    dram_count = 0
    l1_usage = 0
    diff_addresses = []
    global saved_address
    for buffer_info in device_info:
        if buffer_info.buffer_type == ttnn.BufferType.L1:
            l1_usage += buffer_info.max_size_per_bank
        elif buffer_info.buffer_type == ttnn.BufferType.DRAM:
            dram_usage += buffer_info.max_size_per_bank
            dram_count += 1
            if save_address:
                saved_address.append(buffer_info.address)
            elif diff_address:
                if buffer_info.address not in saved_address:
                    diff_addresses.append({"addr": buffer_info.address, "size": buffer_info.max_size_per_bank})
    print("DRAM usage (MB):", dram_usage / 2**20, "across ", dram_count, "buffers")
    print("L1 usage (KB):", l1_usage / 2**10)
    if diff_address:
        print("Difference in addresses:", diff_addresses)
