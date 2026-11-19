# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Test-only fragmentation hypothesis: reserve the real sparse down output.

All logical1..32 views retain one80MiB BF16 tiled buffer. The unchanged native
sparse operation zeros the expanded optional output on every call (nnz=None).
Only the selected real down-weight buffer is intercepted. This is deliberately
not a runtime change or a supported public-decoder workspace API.
"""

import hashlib
import json
from pathlib import Path
from unittest.mock import patch


def _source_sha256():
    root = Path(__file__).resolve().parents[4]
    files = [
        Path(__file__),
        root / "ttnn/cpp/ttnn/operations/matmul/device/sparse/sparse_matmul_device_operation.cpp",
        root / "ttnn/cpp/ttnn/operations/data_movement/reshape_view/reshape.cpp",
        root / "ttnn/cpp/ttnn/operations/data_movement/reshape_view/codegen/reshape_codegen_supported.cpp",
        root / "ttnn/core/tensor/tensor_ops.cpp",
    ]
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}


class ReservedSparseDownWorkspace:
    def __init__(self, mesh, down_weight, *, logical_counts=None, source_sha256=None):
        import ttnn

        self.ttnn = ttnn
        self.original = ttnn.sparse_matmul
        self.weight_address = down_weight.buffer_address()
        self.weight_dtype = down_weight.dtype
        if tuple(down_weight.shape) != (1, 512, 640, 2560):
            raise ValueError("Expected real target down weight [1,512,640,2560]")
        counts = list(range(1, 33)) if logical_counts is None else list(logical_counts)
        if not counts or len(set(counts)) != len(counts) or any(count not in range(1, 33) for count in counts):
            raise ValueError("Require distinct logical counts1..32")
        self.base = ttnn.empty(
            (1, 512, 32, 2560),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        before = ttnn.get_memory_view(mesh, ttnn.BufferType.DRAM).total_bytes_allocated_per_bank
        self.views = {count: ttnn.reshape(self.base, (1, 512, count, 2560), (1, 512, 32, 2560)) for count in counts}
        after = ttnn.get_memory_view(mesh, ttnn.BufferType.DRAM).total_bytes_allocated_per_bank
        if before != after:
            raise ValueError("Logical views unexpectedly allocate device DRAM")
        for count, view in self.views.items():
            if (
                view.buffer_address() != self.base.buffer_address()
                or tuple(view.shape) != (1, 512, count, 2560)
                or tuple(view.padded_shape) != (1, 512, 32, 2560)
            ):
                raise ValueError("Requested logical view does not retain exact physical storage")
        self.report = {
            "enabled": True,
            "experimental_test_only": True,
            "storage_bytes": 83886080,
            "padded_shape": [1, 512, 32, 2560],
            "logical_counts": counts,
            "input_logical_ranks": [4, 6],
            "output_logical_rank": 4,
            "rank_contract": "Rank4 control or observed rank6 input; both A/B sparse selects B batch axes, giving native output [1,512,Q,2560] for both input ranks",
            "dtype": "bfloat16",
            "memory": "interleaved DRAM",
            "buffer_address": self.base.buffer_address(),
            "weight_address": self.weight_address,
            "all_views_share_address": True,
            "view_allocated_bytes_per_bank": after - before,
            "calls_by_count": {},
            "calls_by_rank_count": {},
            "last_call": None,
            "calls_scope": "Python dispatches including warm/capture; trace replays do not invoke this host wrapper",
            "zeroing": "Unchanged native SparseMatmulDeviceOperation::create_output_tensors calls zeros_like with the same optional output each invocation; nnz=None expanded path",
            "lifetime": "Base and all views retained through device close; no view force-deallocation; sequential CQ0 only; simultaneous consumers unsupported",
            "source_sha256": _source_sha256() if source_sha256 is None else source_sha256,
        }

    def intercept(self):
        return patch.object(self.ttnn, "sparse_matmul", self.call)

    def call(self, left, right, *args, **kwargs):
        if right.buffer_address() != self.weight_address:
            return self.original(left, right, *args, **kwargs)
        count = left.shape[-2]
        rank = len(left.shape)

        # Metadata only: record the actual producer contract before rejecting it.
        # No tensor read, allocation, shape coercion, or native invocation here.
        def metadata(value):
            if hasattr(value, "padded_shape"):
                return {
                    "shape": list(value.shape),
                    "padded_shape": list(value.padded_shape),
                    "dtype": str(value.dtype),
                    "layout": str(value.layout),
                    "address": value.buffer_address(),
                }
            return repr(value)

        self.report["last_call"] = {
            "count": count,
            "rank": rank,
            "phase": "before_validation",
            "left": metadata(left),
            "right": metadata(right),
            "args": [metadata(value) for value in args],
            "kwargs": {name: metadata(value) for name, value in kwargs.items()},
        }
        if (
            args
            or rank not in (4, 6)
            or not 1 <= count <= 32
            or count not in self.views
            or tuple(left.shape) != (1,) * (rank - 3) + (512, count, 640)
            or tuple(left.padded_shape) != (1,) * (rank - 3) + (512, 32, 640)
        ):
            raise ValueError(
                "Unexpected down-projection operand shape/signature: "
                + json.dumps(self.report["last_call"], sort_keys=True)
            )
        if left.dtype != self.ttnn.bfloat16 or left.layout != self.ttnn.TILE_LAYOUT or right.dtype != self.weight_dtype:
            raise ValueError("Unexpected down-projection precision/layout")
        if (
            kwargs.get("nnz") is not None
            or kwargs.get("is_input_a_sparse") is not True
            or kwargs.get("is_input_b_sparse", True) is not True
            or kwargs.get("indices") is not None
            or kwargs.get("optional_output_tensor") is not None
        ):
            raise ValueError("Workspace only supports existing expanded input-A-sparse path")
        if kwargs.get("dtype") != self.ttnn.bfloat16 or kwargs.get("memory_config") != self.ttnn.DRAM_MEMORY_CONFIG:
            raise ValueError("Workspace cannot alter output datatype or placement")
        output = self.views[count]
        if left.buffer_address() == output.buffer_address() or right.buffer_address() == output.buffer_address():
            raise ValueError("Output workspace aliases an operand")
        view = self.ttnn.get_memory_view(left.device(), self.ttnn.BufferType.DRAM)
        self.report["last_call"].update(
            phase="validated",
            free_bytes_per_bank=view.total_bytes_free_per_bank,
            largest_contiguous_bytes_per_bank=view.largest_contiguous_bytes_free_per_bank,
            reserved_output_address=output.buffer_address(),
        )
        kwargs["optional_output_tensor"] = output
        result = self.original(left, right, **kwargs)
        if (
            result.buffer_address() != output.buffer_address()
            or tuple(result.shape) != tuple(output.shape)
            or result.dtype != self.ttnn.bfloat16
        ):
            raise ValueError("Native sparse operation did not use exact reserved output")
        key = str(count)
        self.report["calls_by_count"][key] = self.report["calls_by_count"].get(key, 0) + 1
        key = f"{rank}:{count}"
        self.report["calls_by_rank_count"][key] = self.report["calls_by_rank_count"].get(key, 0) + 1
        self.report["last_call"]["returned_address_exact"] = True
        return result


class PerCallSparseDownWorkspace:
    """Test-only reservation at MLP entry, after GDN/GR, before router/gate/up."""

    def __init__(self, mesh, original):
        self.mesh = mesh
        self.original = original
        self.report = {
            "enabled": True,
            "experimental_test_only": True,
            "storage_bytes_per_call": 83886080,
            "policy": "Reserve at SparseMoe entry; only current-count output view; native expanded zeroing; release Python references after original MLP returns, no force-deallocation",
            "calls_scope": "Python dispatches including capture; replay follows captured allocation/address plan without this Python wrapper",
            "calls": [],
            "source_sha256": _source_sha256(),
        }

    def __call__(self, value):
        import ttnn

        count = value.shape[-2]
        if tuple(value.shape) != (1, 1, count, 2560) or count not in range(1, 33):
            raise ValueError("Expected target MLP BF16 input[1,1,1..32,2560]")
        if value.dtype != ttnn.bfloat16 or value.layout != ttnn.TILE_LAYOUT:
            raise ValueError("Unexpected target MLP datatype/layout")

        def memory():
            view = ttnn.get_memory_view(self.mesh, ttnn.BufferType.DRAM)
            return {
                "banks": view.num_banks,
                "allocated_bytes": view.total_bytes_allocated_per_bank * view.num_banks,
                "free_bytes_per_bank": view.total_bytes_free_per_bank,
                "largest_contiguous_bytes_per_bank": view.largest_contiguous_bytes_free_per_bank,
            }

        entry = {"count": count, "input_shape": list(value.shape), "before_reservation": memory(), "phase": "reserve"}
        self.report["calls"].append(entry)
        workspace = None
        try:
            workspace = ReservedSparseDownWorkspace(
                self.mesh, self.original.down, logical_counts=(count,), source_sha256=self.report["source_sha256"]
            )
            entry.update(
                buffer_address=workspace.base.buffer_address(),
                output_logical_shape=list(workspace.views[count].shape),
                output_padded_shape=list(workspace.views[count].padded_shape),
                view_allocated_bytes_per_bank=workspace.report["view_allocated_bytes_per_bank"],
                after_reservation=memory(),
                phase="original_mlp",
            )
            with workspace.intercept():
                result = self.original(value)
            entry.update(phase="completed", after_original_mlp=memory())
            return result
        finally:
            if workspace is not None:
                entry["native_down"] = workspace.report["last_call"]
                entry["native_down_calls"] = dict(workspace.report["calls_by_count"])
                del workspace
            # Dropping the wrapper's references is sufficient; never force-free
            # any captured internal allocation or metadata alias.
            entry["after_workspace_reference_release"] = memory()
