# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
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


def _print_memory_stats(device, message=""):
    # return
    # Only log for device 0 in multi-device setup
    if device.id() != 0:
        return
    memory_view = ttnn.device.get_memory_view(device, ttnn.BufferType.DRAM)
    free_per_bank = memory_view.total_bytes_free_per_bank
    cont_free_per_bank = memory_view.largest_contiguous_bytes_free_per_bank
    badly_allocated = free_per_bank - cont_free_per_bank
    logger.debug("-" * 40)
    logger.debug(f"At: '{message}'")
    logger.debug(f"Total bytes per bank: {memory_view.total_bytes_per_bank}")
    logger.debug(f"Allocated per bank: {memory_view.total_bytes_allocated_per_bank}")
    logger.debug(f"Free per bank: {free_per_bank} ({round(free_per_bank / 1e6, 2)} MB)")
    logger.debug(
        f"Largest contiguous free: {cont_free_per_bank} ({round(cont_free_per_bank / 1e6, 2)} MB, {round(100*cont_free_per_bank / free_per_bank)}%)"
    )
    logger.debug(
        f"Badly allocated: {badly_allocated} ({round(badly_allocated / 1e6, 2)} MB, {round(100*badly_allocated / free_per_bank)}%)"
    )
    logger.debug("-" * 40)


def dump_ttnn_meminfo(mesh_device: ttnn.MeshDevice, header: str = "") -> None:
    """Dump DRAM memory usage of the mesh device to the log."""
    dram_view = ttnn.get_memory_view(mesh_device, ttnn.BufferType.DRAM)
    label = f"DRAM ({header})" if header else "DRAM"
    logger.info(_format_memory_view(dram_view, label))
    # TODO: Add L1 memory view.
