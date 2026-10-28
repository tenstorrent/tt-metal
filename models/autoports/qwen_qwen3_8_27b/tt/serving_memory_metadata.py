# SPDX-License-Identifier: Apache-2.0
"""Non-profiling metadata for physical interleaved cache owners."""
import math


def interleaved_metadata_bytes(rows, banks):
    if banks < 1:
        raise ValueError("Positive physical bank count required")
    owners = {}
    root_ids = {}
    for source in rows:
        if source["buffer_type"] != "DRAM" or source["memory_layout"] != "INTERLEAVED":
            raise ValueError("Physical cache census requires interleaved DRAM owners")
        address = source["address"]
        pages, aligned = source["pages"], source["aligned_page_bytes"]
        if pages < 1 or aligned < 1:
            raise ValueError("Allocated buffer page metadata required")
        row = dict(
            address=address,
            pages=pages,
            aligned_page_bytes=aligned,
            physical_bytes_per_device=math.ceil(pages / banks) * banks * aligned,
        )
        if address in owners and owners[address] != row:
            raise ValueError("Aliased cache owner has inconsistent physical page metadata")
        owners[address] = row
        if "root_buffer_unique_id" in source:
            root_ids.setdefault(address, set()).add(source["root_buffer_unique_id"])
    for address, row in owners.items():
        row["reported_root_buffer_unique_ids"] = sorted(root_ids.get(address, ()))
    return dict(
        bytes_per_device=sum(row["physical_bytes_per_device"] for row in owners.values()),
        unique_physical_buffers=len(owners),
        banks=banks,
        basis="Deduplicated live interleaved DRAM root-buffer page metadata; bank padding included",
        owners=list(owners.values()),
    )


def interleaved_owner_bytes(tensors, banks):
    """Use real buffer page metadata; never logical tensor element counts.

    Scope is the model's interleaved DRAM cache tensors. Sharded weights and
    allocator/program/TRACE buffers are deliberately outside this subtotal.
    """
    import ttnn

    rows = []
    for tensor in tensors:
        config = tensor.memory_config()
        if config.buffer_type != ttnn.BufferType.DRAM or config.memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
            raise ValueError("Physical cache census requires interleaved DRAM owners")
        rows.append(
            dict(
                address=tensor.buffer_address(),
                root_buffer_unique_id=tensor.buffer_unique_id(),
                pages=tensor.buffer_num_pages(),
                aligned_page_bytes=tensor.buffer_aligned_page_size(),
                buffer_type="DRAM",
                memory_layout="INTERLEAVED",
            )
        )
    return interleaved_metadata_bytes(rows, banks)


def allocator_views(mesh):
    """Host allocator metadata only: no graph capture, profiler or synchronization."""
    import ttnn

    result = {}
    for name in ("DRAM", "TRACE"):
        view = ttnn.get_memory_view(mesh, getattr(ttnn.BufferType, name))
        result[name] = dict(
            banks=view.num_banks,
            capacity_bytes_per_device=view.total_bytes_per_bank * view.num_banks,
            allocated_bytes_per_device=view.total_bytes_allocated_per_bank * view.num_banks,
            free_bytes_per_device=view.total_bytes_free_per_bank * view.num_banks,
            largest_contiguous_free_bytes_per_bank=view.largest_contiguous_bytes_free_per_bank,
        )
    return result
