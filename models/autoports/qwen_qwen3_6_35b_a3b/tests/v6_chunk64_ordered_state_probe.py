# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Isolated chunk64 diagnostic with explicit canonical triangular sum ordering.

Imports the unchanged base component. Only its triangular correction reduction
is replaced. This is a device candidate, not a claim of device bit exactness.
"""

import argparse
import ast
import hashlib
import json
from pathlib import Path

import torch

from models.autoports.qwen_qwen3_6_35b_a3b.tests import v6_chunk64_state_probe as base


class OrderedChunk64StateGraph(base.Chunk64StateGraph):
    sum_order = "canonical"

    def triangular_sum(self, products):
        """Sum [..., i, i] to [..., 1, i], matching proven CPU add scheduling."""
        t = self.t
        shape = tuple(products.shape)
        rows, columns = shape[-2:]
        assert len(shape) == 4 and rows == columns and 1 <= rows <= 63
        if self.sum_order == "native":
            return t.sum(products, dim=-2, keepdim=True, compute_kernel_config=self.compute, memory_config=self.mem)
        assert self.sum_order == "canonical"

        def zeros(width):
            return t.zeros(
                (shape[0], shape[1], 1, width),
                dtype=t.float32,
                layout=t.TILE_LAYOUT,
                device=self.mesh,
                memory_config=self.mem,
            )

        def row(index, width):
            return t.slice(products, (0, 0, index, 0), (shape[0], shape[1], index + 1, width), memory_config=self.mem)

        partials = [zeros(columns) for _ in range(4)]
        end = rows // 4 * 4
        for index in range(end):
            slot = index % 4
            partials[slot] = self.add(partials[slot], row(index, columns))
        for index in range(end, rows):
            partials[0] = self.add(partials[0], row(index, columns))
        for index in range(1, 4):
            partials[0] = self.add(partials[0], partials[index])
        result = self.add(zeros(columns), partials[0])

        cascade_columns = columns // 32 * 32 if columns >= 8 else columns // 4 * 4
        if cascade_columns:
            completed = zeros(cascade_columns)
            end = rows // 16 * 16
            for start in range(0, end, 16):
                partial = zeros(cascade_columns)
                for index in range(start, start + 16):
                    partial = self.add(partial, row(index, cascade_columns))
                completed = self.add(completed, partial)
            tail = zeros(cascade_columns)
            for index in range(end, rows):
                tail = self.add(tail, row(index, cascade_columns))
            prefix = self.add(zeros(cascade_columns), self.add(tail, completed))
            if cascade_columns == columns:
                result = prefix
            else:
                suffix = t.slice(
                    result, (0, 0, 0, cascade_columns), (shape[0], shape[1], 1, columns), memory_config=self.mem
                )
                result = t.concat([prefix, suffix], dim=-1, memory_config=self.mem)
        return result


def attach_call():
    """Copy exact current base __call__ AST, replacing only correction's t.sum."""
    tree = ast.parse(Path(base.__file__).read_text())
    graph = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Chunk64StateGraph")
    method = next(node for node in graph.body if isinstance(node, ast.FunctionDef) and node.name == "__call__")
    replaced = 0
    for node in ast.walk(method):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id == "correction":
                assert isinstance(node.value, ast.Call) and ast.unparse(node.value.func) == "t.sum"
                assert ast.unparse(node.value.args[0]) == "product"
                node.value = ast.parse("self.triangular_sum(product)", mode="eval").body
                replaced += 1
    assert replaced == 1
    module = ast.Module(body=[method], type_ignores=[])
    ast.fix_missing_locations(module)
    scope = dict(vars(base))
    exec(compile(module, str(Path(__file__).resolve()) + ":base_call_with_ordered_sum", "exec"), scope)
    OrderedChunk64StateGraph.__call__ = scope["__call__"]
    return hashlib.sha256(ast.dump(module).encode()).hexdigest()


def cpu_schedule_check():
    """Exercise the same helper through a CPU adapter; never imports TTNN."""

    class Adapter:
        float32, TILE_LAYOUT = torch.float32, "tile"

        @staticmethod
        def zeros(shape, **kw):
            return torch.zeros(shape)

        @staticmethod
        def add(a, b, **kw):
            return a + b

        @staticmethod
        def slice(value, begins, ends, **kw):
            return value[tuple(slice(begin, end) for begin, end in zip(begins, ends))]

        @staticmethod
        def concat(values, dim, **kw):
            return torch.cat(values, dim=dim)

    graph = object.__new__(OrderedChunk64StateGraph)
    graph.t, graph.mesh, graph.mem, graph.sum_order = Adapter, None, None, "canonical"
    generator = torch.Generator().manual_seed(20261113)
    differences, elements = 0, 0
    for width in range(1, 64):
        products = torch.randn(1, 32, width, width, generator=generator)
        expected = products.sum(-2, keepdim=True)
        actual = graph.triangular_sum(products)
        differences += int((expected.contiguous().view(torch.int32) != actual.contiguous().view(torch.int32)).sum())
        elements += expected.numel()
    assert differences == 0
    return dict(cases=63, fp32_elements=elements, fp32_code_differences=differences, backend="CPU adapter, not TTNN")


def run(args):
    base_hash = base.sha(base.__file__)
    wrapper_hash = base.sha(__file__)
    call_hash = attach_call()
    old_graph, old_setup = base.Chunk64StateGraph, base.cpu_setup
    OrderedChunk64StateGraph.sum_order = args.sum_order

    def setup(options):
        result, payload = old_setup(options)
        result.update(
            sum_order=args.sum_order,
            ordered_sum_wrapper=str(Path(__file__).resolve()),
            ordered_sum_wrapper_sha256=wrapper_hash,
            unchanged_base_source_sha256=base_hash,
            patched_call_ast_sha256=call_hash,
            helper_cpu_schedule_proof=cpu_schedule_check(),
            triangular_sum_claim="CPU schedule exact; device FP32 add and slice behavior still require measurement",
        )
        snapshot = Path(str(args.output) + ".ordered_sum_wrapper.py.snapshot")
        snapshot.write_bytes(Path(__file__).read_bytes())
        result["ordered_sum_wrapper_snapshot"] = str(snapshot)
        helper = ast.parse(Path(__file__).read_text())
        cls = next(
            node for node in helper.body if isinstance(node, ast.ClassDef) and node.name == "OrderedChunk64StateGraph"
        )
        calls = [ast.unparse(node.func) for node in ast.walk(cls) if isinstance(node, ast.Call)]
        assert not any(call.startswith("torch.") or call.endswith((".from_torch", ".to_torch")) for call in calls)
        result["ordered_sum_runtime_ast_audit"] = "no torch/from_torch/to_torch calls"
        return result, payload

    base.Chunk64StateGraph, base.cpu_setup = OrderedChunk64StateGraph, setup
    try:
        base.run(args)
    finally:
        base.Chunk64StateGraph, base.cpu_setup = old_graph, old_setup
        assert base.sha(base.__file__) == base_hash and base.sha(__file__) == wrapper_hash
        provenance = Path(str(args.output) + ".provenance.json")
        if provenance.exists():
            data = json.loads(provenance.read_text())
            data.update(
                ordered_sum_wrapper_sha256=wrapper_hash,
                unchanged_base_source_sha256=base_hash,
                patched_call_ast_sha256=call_hash,
                sum_order=args.sum_order,
                acceptance_eligible=False,
            )
            provenance.write_text(json.dumps(data, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lengths", type=int, nargs="+", default=[128])
    parser.add_argument("--initial-state", choices=["zero", "prefix64"], default="zero")
    parser.add_argument("--variant", choices=["native", "sfpu_reduce", "sfpu_fma"], default="sfpu_fma")
    parser.add_argument("--cumsum", choices=["plain", "compensated"], default="compensated")
    parser.add_argument("--sum-order", choices=["native", "canonical"], default="canonical")
    parser.add_argument("--oracle-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
