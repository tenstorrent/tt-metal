# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Test-only native sum without cloning an exclusively owned padded input."""

from unittest.mock import patch


def sum_private_down(value, *, native_sum=None, record=None):
    import ttnn

    count = value.shape[-2]
    if (
        tuple(value.shape) != (1, 512, count, 2560)
        or count not in range(1, 33)
        or tuple(value.padded_shape) != (1, 512, 32, 2560)
        or value.dtype != ttnn.bfloat16
        or value.layout != ttnn.TILE_LAYOUT
        or value.memory_config() != ttnn.DRAM_MEMORY_CONFIG
    ):
        raise ValueError("Expected exclusively owned native BF16 down[1,512,Q,2560], paddedQ32")
    call = ttnn.sum if native_sum is None else native_sum
    record = {} if record is None else record

    def allocated():
        view = ttnn.get_memory_view(value.device(), ttnn.BufferType.DRAM)
        return view.total_bytes_allocated_per_bank * view.num_banks

    record.update(count=count, input_address=value.buffer_address(), input_shape=list(value.shape))
    if count == 32:
        record["aligned_original_path"] = True
        return call(value, dim=1, keepdim=True)
    before = allocated()
    filled = ttnn.fill_implicit_tile_padding(value, 0.0)
    after_fill = allocated()
    aligned = ttnn.reshape(filled, (1, 512, 32, 2560), (1, 512, 32, 2560))
    after_view = allocated()
    if filled.buffer_address() != value.buffer_address() or aligned.buffer_address() != value.buffer_address():
        raise ValueError("Private padding fill/physical-height view must alias original buffer")
    record.update(
        fill_address=filled.buffer_address(),
        view_address=aligned.buffer_address(),
        fill_allocated_delta=after_fill - before,
        view_allocated_delta=after_view - after_fill,
    )
    output = call(aligned, dim=1, keepdim=True)
    if tuple(output.shape) != (1, 1, 32, 2560) or output.dtype != ttnn.bfloat16:
        raise ValueError("Unexpected native aligned reduction result")
    result = ttnn.reshape(output, (1, 1, count, 2560), (1, 1, 32, 2560))
    if result.buffer_address() != output.buffer_address():
        raise ValueError("Restoring logicalQ must be a metadata-only output view")
    record.update(
        output_address=result.buffer_address(), output_shape=list(result.shape), same_native_sum_configuration=True
    )
    return result


class ReusePrivateReductionBuffer:
    def __init__(self, original):
        self.original = original
        self.down = original.down
        self.report = {
            "enabled": True,
            "test_only": True,
            "policy": "Only actual native sparse down buffer; in-place zero padding, same-storage Q32 view, unchanged public sum, logicalQ output view",
            "calls": [],
        }

    def __call__(self, value):
        import ttnn

        native_sparse, native_sum = ttnn.sparse_matmul, ttnn.sum
        address = None
        consumed = False
        entry = {"count": value.shape[-2]}
        self.report["calls"].append(entry)

        def sparse(left, right, *args, **kwargs):
            nonlocal address
            result = native_sparse(left, right, *args, **kwargs)
            if right.buffer_address() == self.down.buffer_address():
                if address is not None:
                    raise ValueError("Multiple private down producers")
                address = result.buffer_address()
            return result

        def reduce(tensor, *args, **kwargs):
            nonlocal consumed
            if address is None or tensor.buffer_address() != address or consumed:
                return native_sum(tensor, *args, **kwargs)
            if args or kwargs != {"dim": 1, "keepdim": True}:
                raise ValueError("Unexpected private down sum signature")
            consumed = True
            return sum_private_down(tensor, native_sum=native_sum, record=entry)

        with patch.object(ttnn, "sparse_matmul", sparse), patch.object(ttnn, "sum", reduce):
            result = self.original(value)
        if not consumed:
            raise ValueError("Private down reduction was not intercepted")
        return result


def allocation_summary(graph, banks):
    """Actual NORMAL graph allocation events, not estimated tensor sizes."""
    live, peak, allocations, events = {}, 0, [], []
    for node in graph:
        kind, params = node["node_type"], node.get("params", {})
        if kind not in ("buffer_allocate", "buffer_deallocate") or params.get("type") != "DRAM":
            continue
        key = (int(params["device_id"]), int(params["address"]))
        size = int(params["size"])
        aligned = int(params["max_size_per_bank"]) * banks
        if kind == "buffer_allocate":
            if key in live:
                raise ValueError("Duplicate live DRAM allocation in capture")
            live[key] = aligned
            allocations.append(size)
        else:
            if key not in live:
                raise ValueError("Graph deallocated an untracked preexisting buffer")
            del live[key]
        peak = max(peak, sum(live.values()))
        events.append({"kind": kind, "address": key[1], "size": size, "aligned_bytes": aligned})
    if not allocations:
        raise ValueError("Graph has no DRAM allocation evidence")
    return {
        "additional_peak_allocated_bytes": peak,
        "allocation_sizes": allocations,
        "events": events,
        "contains_80mib_allocation": 83886080 in allocations,
        "scope": "Allocator events during NORMAL dispatched graph; additional live DRAM above preexisting inputs, aligned across banks",
    }
