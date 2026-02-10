# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from loguru import logger

import ttnn


def _bytes_to_mib(num_bytes: int) -> float:
    return num_bytes / (1024 * 1024)


def _format_memory_view(view: ttnn._ttnn.device.MemoryView, label: str) -> str:
    total_bytes = view.total_bytes_per_bank * view.num_banks
    allocated_bytes = view.total_bytes_allocated_per_bank * view.num_banks
    free_bytes = view.total_bytes_free_per_bank * view.num_banks
    percent_used = (allocated_bytes / total_bytes * 100.0) if total_bytes else 0.0
    total_mib = _bytes_to_mib(total_bytes)
    allocated_mib = _bytes_to_mib(allocated_bytes)
    free_mib = _bytes_to_mib(free_bytes)
    allocated_per_bank_mib = _bytes_to_mib(view.total_bytes_allocated_per_bank)
    free_per_bank_mib = _bytes_to_mib(view.total_bytes_free_per_bank)
    largest_contig_mib = _bytes_to_mib(view.largest_contiguous_bytes_free_per_bank)
    per_bank_mib = _bytes_to_mib(view.total_bytes_per_bank)
    return (
        f"{label} usage: {allocated_mib:.2f} / {total_mib:.2f} MiB "
        f"({percent_used:.2f}%), free={free_mib:.2f} MiB, "
        f"largest_contiguous_free_per_bank={largest_contig_mib:.2f} MiB, "
        f"banks={view.num_banks}, per_bank={per_bank_mib:.2f} MiB, "
        f"allocated_per_bank={allocated_per_bank_mib:.2f} MiB, "
        f"free_per_bank={free_per_bank_mib:.2f} MiB"
    )


def dump_ttnn_meminfo(mesh_device: ttnn.MeshDevice, header: str = "") -> None:
    """Dump DRAM memory usage of the mesh device to the log."""
    dram_view = ttnn.get_memory_view(mesh_device, ttnn.BufferType.DRAM)
    logger.info(_format_memory_view(dram_view, f"DRAM ({header})"))
    # TODO: Add L1 memory view.
