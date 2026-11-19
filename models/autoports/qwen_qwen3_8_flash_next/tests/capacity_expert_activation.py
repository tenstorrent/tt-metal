# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Test-only exact private expert-gate reuse, selected by actual weight identity."""

from unittest.mock import patch


class InplaceExpertActivation:
    def __init__(self, mesh, original):
        self.mesh = mesh
        self.original = original
        self.down = original.down  # Supports the independent per-call down reservation.
        self.gate_address = original.gate.buffer_address()
        self.up_address = original.up.buffer_address()
        self.down_address = original.down.buffer_address()
        self.report = {
            "enabled": True,
            "test_only": True,
            "policy": "Only native sparse gate/up outputs identified by actual expert weight addresses; silu(gate,out=gate), multiply(gate,up,out=gate); shared expert untouched",
            "calls_scope": "Python eager/capture dispatches; replay executes captured addresses",
            "calls": [],
        }

    def __call__(self, value):
        import ttnn

        count = value.shape[-2]
        if tuple(value.shape) != (1, 1, count, 2560) or count not in range(1, 33):
            raise ValueError("Expected exact real target MLP input shape")
        native_sparse, native_silu, native_multiply = ttnn.sparse_matmul, ttnn.silu, ttnn.multiply
        active = {}
        entry = {"count": count, "operations": [], "phase": "gate_up"}
        self.report["calls"].append(entry)

        def memory():
            view = ttnn.get_memory_view(self.mesh, ttnn.BufferType.DRAM)
            return {
                "allocated_bytes": view.total_bytes_allocated_per_bank * view.num_banks,
                "free_bytes_per_bank": view.total_bytes_free_per_bank,
                "largest_contiguous_bytes_per_bank": view.largest_contiguous_bytes_free_per_bank,
            }

        def sparse(left, right, *args, **kwargs):
            address = right.buffer_address()
            if address == self.down_address:
                if entry["phase"] != "activated" or left.buffer_address() != active.get("gate"):
                    raise ValueError("Native down did not consume the in-place private gate activation")
                entry["down_input_address"] = left.buffer_address()
                active.clear()  # Later shared-expert operations cannot match released addresses.
                entry["phase"] = "down"
            result = native_sparse(left, right, *args, **kwargs)
            if address in (self.gate_address, self.up_address):
                if tuple(result.shape) != (1, 1, 1, 512, count, 640) or tuple(result.padded_shape) != (
                    1,
                    1,
                    1,
                    512,
                    32,
                    640,
                ):
                    raise ValueError("Unexpected real sparse gate/up output shape")
                if result.dtype != ttnn.bfloat16 or result.layout != ttnn.TILE_LAYOUT:
                    raise ValueError("Expected native BF16 tiled expert gate/up")
                name = "gate" if address == self.gate_address else "up"
                if name in active:
                    raise ValueError("Duplicate expert gate/up producer")
                active[name] = result.buffer_address()
                entry[name] = {
                    "address": result.buffer_address(),
                    "logical_shape": list(result.shape),
                    "padded_shape": list(result.padded_shape),
                    "physical_bytes": 20971520,
                }
            return result

        def inplace(call, left, other, operation):
            before = memory()
            result = call(left, *other, output_tensor=left)
            after = memory()
            if (
                result.buffer_address() != left.buffer_address()
                or tuple(result.shape) != tuple(left.shape)
                or result.dtype != left.dtype
            ):
                raise ValueError("Elementwise operation did not return exact private gate alias")
            entry["operations"].append(
                {
                    "operation": operation,
                    "before": before,
                    "after": after,
                    "allocated_delta_bytes": after["allocated_bytes"] - before["allocated_bytes"],
                    "input_output_address": result.buffer_address(),
                    "exact_alias": True,
                }
            )
            return result

        def silu(left, *args, **kwargs):
            if active.get("gate") != left.buffer_address():
                return native_silu(left, *args, **kwargs)
            if args or kwargs or set(active) != {"gate", "up"} or entry["phase"] != "gate_up":
                raise ValueError("Unexpected expert SiLU signature or producer ordering")
            result = inplace(native_silu, left, (), "silu")
            entry["phase"] = "silu"
            return result

        def multiply(left, right, *args, **kwargs):
            left_address = left.buffer_address() if hasattr(left, "buffer_address") else None
            right_address = right.buffer_address() if hasattr(right, "buffer_address") else None
            if not active or left_address != active.get("gate") or right_address != active.get("up"):
                return native_multiply(left, right, *args, **kwargs)
            if args or kwargs or entry["phase"] != "silu":
                raise ValueError("Unexpected expert activation multiply signature")
            result = inplace(native_multiply, left, (right,), "multiply")
            entry["phase"] = "activated"
            return result

        with patch.object(ttnn, "sparse_matmul", sparse), patch.object(ttnn, "silu", silu), patch.object(
            ttnn, "multiply", multiply
        ):
            result = self.original(value)
        if entry["phase"] != "down" or len(entry["operations"]) != 2:
            raise ValueError("Expected exactly two guarded in-place expert activation operations")
        entry["phase"] = "complete"
        return result
