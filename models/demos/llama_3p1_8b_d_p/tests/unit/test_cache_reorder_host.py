# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only cache ordering, tile ownership and bounded-dispatch regressions.

Only the production method is compiled; TTNN and model imports are not executed.
Opaque tiles model storage ownership, not device arithmetic or current kernel aliasing.
"""

import ast
import gc
import math
import unittest
from pathlib import Path
from types import SimpleNamespace

ATTENTION = Path(__file__).resolve().parents[2] / "tt" / "attention.py"


class Storage:
    def __init__(self, pages, borrowed=False):
        self.pages = tuple(pages)
        self.borrowed = borrowed
        self.alive = True
        self.references = 0


class Tensor:
    def __init__(self, shape, storage):
        self.shape = tuple(shape)
        self.storage = storage
        storage.references += 1

    @property
    def pages(self):
        assert self.storage.alive, "use after free"
        return self.storage.pages

    def deallocate(self, force):
        assert force is True
        assert not self.storage.borrowed, "force-freed borrowed gather buffer"
        self.storage.alive = False

    def __del__(self):
        self.storage.references -= 1
        if self.storage.references == 0:
            self.storage.alive = False


def tensor(shape, pages, borrowed=False):
    assert math.prod(shape) // 1024 == len(pages)
    return Tensor(shape, Storage(pages, borrowed))


def chronological(capacity, slot=0):
    # Each unique opaque page names its first absolute row and column, independent of rank packing.
    return tuple((slot, row, col) for row in range(0, capacity, 32) for col in range(0, 128, 32))


def packed(capacity, slot=0):
    return tuple(
        (slot, row, col)
        for rank in range(4)
        for row in range(0, capacity, 32)
        if (row // 256) % 4 == rank
        for col in range(0, 128, 32)
    )


class Backend:
    def __init__(self, no_transpose=False, fail=None):
        self.calls = []
        self.no_transpose = no_transpose
        self.fail = fail
        self.experimental = SimpleNamespace(high_bw_all_gather=self.gather)
        self.owned = []

    def make(self, shape, pages):
        result = tensor(shape, pages)
        self.owned.append(result.storage)
        return result

    def gather(self, cache, **kwargs):
        self.calls.append(("gather", kwargs.copy()))
        output = kwargs["output_tensor"]
        output.storage.pages = cache[kwargs["input_batch_index"]]
        return output

    def reshape(self, value, shape):
        self.calls.append(("reshape", tuple(shape)))
        assert value.shape[-1] == shape[-1] == 128
        assert shape[-2] % 32 == 0 and shape[-1] % 32 == 0
        assert math.prod(value.shape) == math.prod(shape)
        _ = value.pages
        return Tensor(shape, value.storage)

    def transpose(self, value, first, second):
        self.calls.append(("transpose", first, second))
        assert (first, second) == (0, 1), "only the audited CN path is supported"
        if self.fail == "transpose":
            raise RuntimeError("injected transpose failure")
        n, c, h, w = value.shape
        if self.no_transpose:
            # Preserve output allocation/shape so this mutation isolates only incorrect ordering.
            return self.make((c, n, h, w), value.pages)
        if min(n, c) == 1:
            # Model a legal value-preserving view optimization to challenge ownership.
            return Tensor((c, n, h, w), value.storage)
        size = h * w // 1024
        pages = tuple(
            page
            for channel in range(c)
            for batch in range(n)
            for page in value.pages[(batch * c + channel) * size : (batch * c + channel + 1) * size]
        )
        return self.make((c, n, h, w), pages)

    def clone(self, value):
        self.calls.append(("clone",))
        if self.fail == "clone":
            raise RuntimeError("injected clone failure")
        return self.make(value.shape, value.pages)

    def slice(self, value, start, end):
        self.calls.append(("slice", tuple(start), tuple(end)))
        if self.fail == "slice":
            raise RuntimeError("injected slice failure")
        assert value.shape[:2] == (1, 1)
        assert start[:2] == [0, 0] and end[:2] == [1, 1]
        assert start[3] == 0 and end[3] == 128
        assert start[2] % 32 == 0 and end[2] % 32 == 0
        return self.make((1, 1, end[2] - start[2], 128), value.pages[start[2] // 32 * 4 : end[2] // 32 * 4])

    def concat(self, values, dim):
        self.calls.append(("concat", len(values), dim))
        assert dim == 2
        return self.make(
            (1, 1, sum(v.shape[2] for v in values), 128), tuple(page for value in values for page in value.pages)
        )


def method_node(path):
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "FullCausalAttention")
    return next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_gather_and_reorder")


def load_method(backend):
    node = method_node(ATTENTION)
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    namespace = {
        "ttnn": backend,
        "_SP": 4,
        "_SP_AXIS": 0,
        "_GLOBAL_CHUNK": 1024,
        "_LOCAL_SEQUENCE": 256,
        "_HEAD_DIM": 128,
    }
    exec(compile(module, str(ATTENTION), "exec"), namespace)
    return namespace["_gather_and_reorder"]


def setup(capacity, backend=None):
    backend = backend or Backend()
    context = SimpleNamespace(
        max_seq_len=capacity,
        geometry=SimpleNamespace(
            gather_block_order=tuple(
                rank * (capacity // 1024) + chunk for chunk in range(capacity // 1024) for rank in range(4)
            )
        ),
    )
    gather = tensor((1, 1, capacity, 128), chronological(capacity), borrowed=True)
    cache = {31: packed(capacity, 0), 63: packed(capacity, 1)}
    return backend, context, gather, cache, load_method(backend)


class ReorderTests(unittest.TestCase):
    # All supported scale boundaries preserve every opaque tile in chronological order.
    def test_full_order_and_tile_shapes(self):
        for capacity in (1024, 2048, 4096, 8192, 65536, 131072):
            with self.subTest(capacity=capacity):
                backend, context, gather, cache, run = setup(capacity)
                result = run(context, cache, gather, batch_index=31, logical_n=capacity)
                gc.collect()
                self.assertEqual(result.shape, (1, 1, capacity, 128))
                self.assertEqual(result.pages, chronological(capacity))
                result.deallocate(True)
                self.assertTrue(gather.storage.alive)

    # Prefixes immediately before and after tile, SP-block and chunk boundaries exclude future tiles.
    def test_partial_prefixes(self):
        for capacity in (1024, 2048, 8192, 131072):
            for actual_end in (1, 31, 32, 33, 223, 255, 256, 257, 991, 1023, 1024, 1025, capacity - 1, capacity):
                if actual_end > capacity:
                    continue
                length = ((actual_end + 31) // 32) * 32
                with self.subTest(capacity=capacity, length=length):
                    backend, context, gather, cache, run = setup(capacity)
                    result = run(context, cache, gather, batch_index=63, logical_n=length)
                    self.assertEqual(result.shape, (1, 1, length, 128))
                    self.assertEqual(result.pages, chronological(length, 1))
                    result.deallocate(True)
                    self.assertTrue(gather.storage.alive)

    # Reusing the persistent gather for another slot cannot mutate an earlier returned result.
    def test_two_slots_repeat_and_returned_alias_lifetime(self):
        for capacity in (1024, 8192):
            with self.subTest(capacity=capacity):
                backend, context, gather, cache, run = setup(capacity)
                prefix = min(1056, capacity - 32)
                first = run(context, cache, gather, batch_index=31, logical_n=capacity)
                saved = first.pages
                second = run(context, cache, gather, batch_index=63, logical_n=prefix)
                again = run(context, cache, gather, batch_index=31, logical_n=capacity)
                gc.collect()
                self.assertEqual(first.pages, saved)
                self.assertEqual(second.pages, chronological(prefix, 1))
                self.assertEqual(again.pages, first.pages)
                self.assertEqual(len({id(v.storage) for v in (first, second, again)}), 3)
                for value in (first, second, again):
                    value.deallocate(True)
                self.assertTrue(gather.storage.alive)

    # Per-stripe slicing scales with capacity; the optimized path must have a fixed operation bound.
    def test_operation_count_is_bounded(self):
        counts = []
        for capacity in (2048, 8192, 65536, 131072):
            with self.subTest(capacity=capacity):
                backend, context, gather, cache, run = setup(capacity)
                result = run(context, cache, gather, batch_index=31, logical_n=capacity - 32)
                names = [call[0] for call in backend.calls]
                counts.append(len(names))
                self.assertLessEqual(len(names), 6)
                self.assertLessEqual(names.count("slice"), 1)
                self.assertNotIn("concat", names)
                self.assertEqual(result.pages, chronological(capacity - 32))
                result.deallocate(True)
        self.assertEqual(max(counts), min(counts))

    # A missing axis exchange is detected even when shape, volume and finite tile payloads remain valid.
    def test_noop_transpose_mutation_is_detected(self):
        backend, context, gather, cache, run = setup(8192, Backend(no_transpose=True))
        result = run(context, cache, gather, batch_index=31, logical_n=1056)
        self.assertNotEqual(result.pages, chronological(1056))

    # Failure during either new allocation or prefix slicing must not free the borrowed gather.
    def test_failure_keeps_borrowed_buffer_alive(self):
        for capacity, failure in ((1024, "clone"), (1024, "slice"), (8192, "transpose"), (8192, "slice")):
            with self.subTest(capacity=capacity, failure=failure):
                backend, context, gather, cache, run = setup(capacity, Backend(fail=failure))
                with self.assertRaisesRegex(RuntimeError, "injected"):
                    run(context, cache, gather, batch_index=31, logical_n=min(1056, capacity - 32))
                gc.collect()
                self.assertTrue(gather.storage.alive)
                self.assertTrue(all(not storage.alive for storage in backend.owned))


if __name__ == "__main__":
    unittest.main(verbosity=2)
