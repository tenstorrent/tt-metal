# SPDX-License-Identifier: Apache-2.0
"""Per-device full-stack persistent allocation accounting."""
import math

import ttnn


def memory_views(mesh):
    result = {}
    for label, kind in (("dram", ttnn.BufferType.DRAM), ("l1", ttnn.BufferType.L1), ("trace", ttnn.BufferType.TRACE)):
        view = ttnn.get_memory_view(mesh, kind)
        result[label] = {
            key: int(getattr(view, key))
            for key in (
                "num_banks",
                "total_bytes_per_bank",
                "total_bytes_allocated_per_bank",
                "total_bytes_free_per_bank",
                "largest_contiguous_bytes_free_per_bank",
            )
        }
        result[label]["allocated_bytes"] = view.num_banks * view.total_bytes_allocated_per_bank
    return result


def persistent_tensors(root):
    seen_objects, allocations = set(), {}

    def visit(value, path):
        if id(value) in seen_objects:
            return
        seen_objects.add(id(value))
        if isinstance(value, ttnn.Tensor):
            if not value.is_allocated():
                return
            memory = str(value.memory_config().buffer_type)
            address = value.buffer_address()
            key = (memory, address)
            shape = list(value.padded_shape)
            elements = math.prod(shape)
            dtype = str(value.dtype)
            if value.dtype == ttnn.bfloat4_b:
                size = elements // 1024 * 576
            elif value.dtype == ttnn.bfloat8_b:
                size = elements // 1024 * 1088
            else:
                size = elements * (2 if value.dtype in (ttnn.bfloat16, ttnn.uint16) else 4)
            if key in allocations:
                allocations[key]["aliases"].append(path)
            else:
                allocations[key] = dict(
                    path=path,
                    aliases=[],
                    padded_shape=shape,
                    dtype=dtype,
                    memory=memory,
                    address=address,
                    tensor_payload_bytes=size,
                )
        elif isinstance(value, dict):
            for key, child in value.items():
                visit(child, f"{path}.{key}")
        elif isinstance(value, (list, tuple)):
            for i, child in enumerate(value):
                visit(child, f"{path}[{i}]")
        elif type(value).__module__.startswith("models.") and hasattr(value, "__dict__"):
            for name, child in vars(value).items():
                visit(child, f"{path}.{name}")

    visit(root, "generator")
    return list(allocations.values())
