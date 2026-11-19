# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Source identity for stage evidence; never hashes result artifacts."""

import hashlib
import math
from pathlib import Path


def source_identity():
    root = Path(__file__).resolve().parents[1]
    names = (
        "model.py",
        "precision.py",
        "optimized_decoder.py",
        "generator.py",
        "bounded_prefill.py",
        "multichip_decoder.py",
        "flat_prefill.py",
        "prefill_trace.py",
    )
    result = {f"tt/{name}": hashlib.sha256((root / "tt" / name).read_bytes()).hexdigest() for name in names}
    native = "ttnn/cpp/ttnn/operations/data_movement/reshape_view/device/"
    for name in ("reshape_tiled_program_factory.cpp", "device/dataflow/reader_reshape_tiled.cpp"):
        result[f"repo/{native}{name}"] = hashlib.sha256((root.parents[2] / native / name).read_bytes()).hexdigest()
    return result


def ccl_inventory(model):
    """Owned buffer inventory; allocator model, separate from measured memory."""
    import ttnn

    tile_bytes = {
        ttnn.bfloat16: 2048,
        ttnn.bfloat8_b: 1088,
        ttnn.float32: 4096,
        ttnn.uint32: 4096,
        ttnn.int32: 4096,
        ttnn.uint16: 2048,
    }
    seen = set()
    layers = []
    for index, layer in zip(model.layer_indices, model.layers):
        size = 0
        count = 0
        for buffers in layer._ccl_buffers.values():
            for tensor in buffers:
                if tensor is None or tensor.buffer_address() in seen:
                    continue
                seen.add(tensor.buffer_address())
                if tensor.memory_config() != ttnn.DRAM_MEMORY_CONFIG or tensor.layout != ttnn.TILE_LAYOUT:
                    raise AssertionError("Selected persistent CCL inventory expects interleaved DRAM tiles")
                shape = tuple(tensor.padded_shape)
                pages = math.prod(shape) // 1024
                size += ((pages + 7) // 8 * 8) * ((tile_bytes[tensor.dtype] + 63) // 64 * 64)
                count += 1
        layers.append(dict(layer=index, signatures=len(layer._ccl_buffers), buffers=count, bytes_per_device=size))
    return dict(
        layers=layers,
        bytes_per_device=sum(row["bytes_per_device"] for row in layers),
        scope="Distinct owned CCL DRAM buffers; source allocator model, not peak or total DRAM",
    )
