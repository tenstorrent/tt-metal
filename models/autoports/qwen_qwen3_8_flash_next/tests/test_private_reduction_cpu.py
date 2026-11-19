# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Allocation evidence accounting without importing TTNN or opening hardware."""

from models.autoports.qwen_qwen3_8_flash_next.tests.capacity_private_reduction import allocation_summary


def test_peak_counts_overlapping_allocations_and_address_reuse():
    def event(kind, address, size):
        return {
            "node_type": kind,
            "params": {
                "type": "DRAM",
                "device_id": 0,
                "address": address,
                "size": size,
                "max_size_per_bank": size // 8,
            },
        }

    graph = [
        event("buffer_allocate", 10, 83886080),
        event("buffer_allocate", 20, 163840),
        event("buffer_deallocate", 10, 83886080),
        event("buffer_allocate", 10, 8192),
    ]
    result = allocation_summary(graph, 8)
    assert result["additional_peak_allocated_bytes"] == 83886080 + 163840
    assert result["contains_80mib_allocation"]
    assert result["allocation_sizes"] == [83886080, 163840, 8192]


def test_padding_candidate_small_output_only():
    graph = [
        {
            "node_type": "buffer_allocate",
            "params": {
                "type": "DRAM",
                "device_id": "0",
                "address": "10",
                "size": "163840",
                "max_size_per_bank": "20480",
            },
        }
    ]
    result = allocation_summary(graph, 8)
    assert result["additional_peak_allocated_bytes"] == 163840
    assert not result["contains_80mib_allocation"]
