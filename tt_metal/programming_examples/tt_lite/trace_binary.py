# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Python implementation of the .ttb (Tenstorrent Trace Binary) file format.
Compatible with the C++ reader in trace_binary.h.
"""

import struct
from dataclasses import dataclass, field

TTB_MAGIC = 0x54544230  # "TTB0"
TTB_VERSION = 0


BUFFER_TYPE_DRAM = 0
BUFFER_TYPE_L1 = 1


@dataclass
class BufferPlacement:
    address: int
    size: int
    page_size: int
    buffer_type: int = BUFFER_TYPE_DRAM


@dataclass
class TraceBinary:
    worker_descs: list = field(default_factory=list)
    trace_streams: list = field(default_factory=list)
    persistent_buffers: list = field(default_factory=list)
    persistent_buffer_data: list = field(default_factory=list)
    io_buffers: list = field(default_factory=list)
    io_buffer_names: list = field(default_factory=list)
    trace_buf_address: int = 0
    trace_buf_page_size: int = 0
    trace_buf_num_pages: int = 0


def write_trace_binary(ttb: TraceBinary, path: str) -> bool:
    try:
        with open(path, "wb") as f:
            # Header
            f.write(
                struct.pack(
                    "<IIIIII",
                    TTB_MAGIC,
                    TTB_VERSION,
                    len(ttb.worker_descs),
                    len(ttb.trace_streams),
                    len(ttb.persistent_buffers),
                    len(ttb.io_buffers),
                )
            )

            # Worker descriptors
            for wd in ttb.worker_descs:
                f.write(struct.pack("<B", wd["sub_device_id"]))
                f.write(struct.pack("<I", wd["num_completion_worker_cores"]))
                f.write(struct.pack("<I", wd["num_mcast_programs"]))
                f.write(struct.pack("<I", wd["num_unicast_programs"]))

            # Trace streams
            for stream in ttb.trace_streams:
                f.write(struct.pack("<I", len(stream)))
                for val in stream:
                    f.write(struct.pack("<I", val))

            # Persistent buffers
            for i, bp in enumerate(ttb.persistent_buffers):
                f.write(struct.pack("<Q", bp.address))
                f.write(struct.pack("<Q", bp.size))
                f.write(struct.pack("<I", bp.page_size))
                f.write(struct.pack("<B", bp.buffer_type))
                data = ttb.persistent_buffer_data[i]
                f.write(struct.pack("<Q", len(data)))
                f.write(data)

            # IO buffers
            for i, bp in enumerate(ttb.io_buffers):
                f.write(struct.pack("<Q", bp.address))
                f.write(struct.pack("<Q", bp.size))
                f.write(struct.pack("<I", bp.page_size))
                f.write(struct.pack("<B", bp.buffer_type))
                name = ttb.io_buffer_names[i].encode("utf-8")
                f.write(struct.pack("<I", len(name)))
                f.write(name)

            # Trace buffer placement
            f.write(struct.pack("<Q", ttb.trace_buf_address))
            f.write(struct.pack("<I", ttb.trace_buf_page_size))
            f.write(struct.pack("<I", ttb.trace_buf_num_pages))

        return True
    except Exception as e:
        print(f"Error writing trace binary: {e}")
        return False


def export_trace(device, trace_id, output_path, io_tensors, persistent_tensors=None):
    """
    Export a captured trace to a .ttb file.

    Args:
        device: ttnn MeshDevice (or Device)
        trace_id: MeshTraceId from ttnn.begin_trace_capture/end_trace_capture
        output_path: path to write the .ttb file
        io_tensors: dict of {name: ttnn.Tensor} for IO buffer metadata
        persistent_tensors: list of ttnn.Tensor (model weights on device) to include in .ttb
    """
    import ttnn

    trace_data = ttnn.get_trace_data(device, trace_id)

    ttb = TraceBinary()

    for wd in trace_data.worker_descs:
        ttb.worker_descs.append(
            {
                "sub_device_id": wd.sub_device_id,
                "num_completion_worker_cores": wd.num_completion_worker_cores,
                "num_mcast_programs": wd.num_mcast_programs,
                "num_unicast_programs": wd.num_unicast_programs,
            }
        )

    for stream in trace_data.trace_streams:
        ttb.trace_streams.append(list(stream))

    ttb.trace_buf_address = trace_data.trace_buf_address
    ttb.trace_buf_page_size = trace_data.trace_buf_page_size
    ttb.trace_buf_num_pages = trace_data.trace_buf_num_pages

    if persistent_tensors:
        total_weight_bytes = 0
        for i, tensor in enumerate(persistent_tensors):
            addr = tensor.buffer_address()
            buf_size = tensor.buffer_page_size() * tensor.buffer_num_pages()
            page_size = tensor.buffer_page_size()
            ttb.persistent_buffers.append(BufferPlacement(address=addr, size=buf_size, page_size=page_size))
            raw_u32 = ttnn.read_raw_buffer_data(device, tensor)
            raw_bytes = struct.pack(f"<{len(raw_u32)}I", *raw_u32)
            ttb.persistent_buffer_data.append(raw_bytes)
            total_weight_bytes += len(raw_bytes)
            if (i + 1) % 50 == 0:
                print(f"  Serialized {i + 1}/{len(persistent_tensors)} weight tensors...")
        print(f"  Total weight data: {total_weight_bytes / (1024 * 1024):.1f} MB ({len(persistent_tensors)} tensors)")

    for name, tensor in io_tensors.items():
        mem_cfg = tensor.memory_config()
        buf_type = BUFFER_TYPE_L1 if mem_cfg.buffer_type == ttnn.BufferType.L1 else BUFFER_TYPE_DRAM
        ttb.io_buffers.append(
            BufferPlacement(
                address=tensor.buffer_address(),
                size=tensor.buffer_page_size() * tensor.buffer_num_pages(),
                page_size=tensor.buffer_page_size(),
                buffer_type=buf_type,
            )
        )
        ttb.io_buffer_names.append(name)

    if write_trace_binary(ttb, output_path):
        print(f"Trace binary written to: {output_path}")
        print(f"  Trace streams: {len(ttb.trace_streams)}")
        for i, stream in enumerate(ttb.trace_streams):
            print(f"    Stream {i}: {len(stream) * 4} bytes")
        print(f"  Trace buffer addr: 0x{ttb.trace_buf_address:x}")
        print(f"  Persistent buffers: {len(ttb.persistent_buffers)}")
        print(f"  IO buffers: {len(ttb.io_buffers)}")
        for i, bp in enumerate(ttb.io_buffers):
            print(f"    {ttb.io_buffer_names[i]} addr=0x{bp.address:x} size={bp.size} page_size={bp.page_size}")
        return True
    return False
