# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only constructor/release integration via the actual generator method AST.

Stub allocations expose ownership metadata; no TTNN/torch import or device work.
This tests binding semantics only, not TT execution or hardware mesh validation.
"""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
from weakref import WeakValueDictionary

from models.autoports.ornith_ai_ornith_1_5_9b.tt.cache_contract import (
    PERSISTENT_CATEGORIES,
    CacheOwnershipRegistry,
    DramBudget,
    get_kv_cache_spec,
    get_max_tokens_all_users,
    plan_cache_allocation,
    validate_host_page_table,
)


class Tensor:
    def __init__(self, shape, dtype, owner):
        self.shape, self.dtype, self.owner = shape, dtype, owner
        self.layout, self.live, self.payload = "Layout.TILE", True, object()

    def device(self):
        return self.owner

    def is_allocated(self):
        return self.live


class Table:
    def __init__(self, rows):
        self.rows = [list(row) for row in rows]
        self.shape = (len(rows), len(rows[0]))
        self.device, self.dtype = NS(type="cpu"), "torch.int32"

    def tolist(self):
        return [row.copy() for row in self.rows]

    def clone(self):
        return Table(self.rows)


class Model:
    def __init__(self):
        self.device = NS(shape=(1, 4), identity=object())
        self.checkpoint, self.config = NS(path="unused"), NS(max_position_embeddings=262144)
        self.layer_indices = [0, 3]
        self.layers = [
            NS(kind="linear_attention", kv_dtype="bfloat8_b"),
            NS(kind="full_attention", kv_dtype="bfloat8_b"),
        ]
        self.vocab_size, self.allocations = 248320, []

    def external(self, blocks=19, slots=2):
        conv = Tensor((3, 1, slots, 2048), "bfloat16", self.device)
        rec = Tensor((slots, 8, 128, 128), "float32", self.device)
        key, value = [Tensor((blocks, 1, 32, 256), "bfloat8_b", self.device) for _ in range(2)]
        common = {"page_offset": 0, "page_count": None, "global_pages": False, "key": None, "value": None}
        linear = [NS(**common, conv=conv, recurrent=rec, state_row=i) for i in range(slots)]
        full = [
            NS(
                conv=None,
                recurrent=None,
                key=key,
                value=value,
                state_row=None,
                page_offset=i * 8,
                page_count=8,
                global_pages=True,
            )
            for i in range(slots)
        ]
        return [linear, full]

    def allocate_cache(self, **kwargs):
        self.allocations.append(kwargs)
        return self.external(blocks=kwargs["batch"] * ((kwargs["context"] + 31) // 32), slots=kwargs["batch"])


class ExternalOwnershipTests(unittest.TestCase):
    def setUp(self):
        self.resets = []
        helper = NS(reset_slots=lambda **kwargs: self.resets.append(kwargs))

        class Sampler:
            def __init__(self, **kwargs):
                pass

            def reset_sampling_params(self, params):
                pass

        class SamplingState:
            def __init__(self, *args, **kwargs):
                pass

            def prepare(self):
                return self

        def arange(count, **kwargs):
            return NS(reshape=lambda rows, cols: Table([range(i * cols, (i + 1) * cols) for i in range(rows)]))

        self.fake_torch = NS(arange=arange, int32="torch.int32")
        self.fake_transformers = NS(AutoTokenizer=NS(from_pretrained=lambda path: object()))
        path = Path(__file__).resolve().parents[1] / "tt/generator.py"
        tree = ast.parse(path.read_text())
        sampling_capacity = next(
            ast.literal_eval(node.value)
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "DEVICE_SAMPLING_MAX_TOP_K" for target in node.targets
            )
        )
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "OrnithGenerator")
        selected = {
            "__init__",
            "bind_serving_cache",
            "get_kv_cache_spec",
            "get_max_tokens_all_users",
            "reset_slots",
            "_invalidate_resident",
            "release_request",
        }
        cls.body = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in selected]
        self.assertEqual({node.name for node in cls.body}, selected)
        namespace = {
            "Generator": object,
            "DEVICE_SAMPLING_MAX_TOP_K": sampling_capacity,
            "SimpleNamespace": NS,
            "WeakValueDictionary": WeakValueDictionary,
            "CacheOwnershipRegistry": CacheOwnershipRegistry,
            "validate_host_page_table": validate_host_page_table,
            "cache_contract": NS(
                get_kv_cache_spec=get_kv_cache_spec, get_max_tokens_all_users=get_max_tokens_all_users
            ),
            "SamplingGenerator": Sampler,
            "SamplingParams": lambda **kwargs: NS(**kwargs),
            "SamplingSlotState": SamplingState,
            "format_sampling_params": lambda p, n, **kwargs: p,
            "SamplingCCL": lambda model: object(),
            "PrefillContract": NS(prepare=lambda *args, **kwargs: helper),
            "ttnn": NS(zeros_like=lambda tensor: Tensor(tensor.shape, tensor.dtype, tensor.owner)),
        }
        exec(  # noqa: S102 -- inspected local diagnostic source
            compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec"), namespace
        )
        self.Generator = namespace["OrnithGenerator"]

    def construct(self, model, **kwargs):
        with patch.dict("sys.modules", {"torch": self.fake_torch, "transformers": self.fake_transformers}):
            return self.Generator(model, max_batch_size=2, max_seq_len=256, **kwargs)

    def test_injected_cache_does_not_allocate_duplicate_or_modify_backing(self):
        model = Model()
        cache, table = model.external(), Table([range(8), range(10, 18)])
        payloads = [
            getattr(st, name).payload
            for states in cache
            for st in states[:1]
            for name in ("conv", "recurrent", "key", "value")
            if getattr(st, name) is not None
        ]
        gen = self.construct(model, kv_cache=cache, page_table=table)
        self.assertEqual(model.allocations, [])
        self.assertIs(gen.cache, cache)
        self.assertTrue(gen.cache_is_external)
        self.assertIsNot(gen.page_table, table)
        self.assertEqual(gen.page_table.tolist(), table.tolist())
        self.assertEqual(
            payloads,
            [
                getattr(st, name).payload
                for states in cache
                for st in states[:1]
                for name in ("conv", "recurrent", "key", "value")
                if getattr(st, name) is not None
            ],
        )
        gen.release_request(1)
        self.assertIs(self.resets[-1]["cache"], cache)
        self.assertFalse(self.resets[-1]["clear_kv"])
        with self.assertRaises(ValueError):
            gen.get_kv_cache_spec()  # Reduced diagnostics cannot advertise full specs.

    def test_default_allocation_and_later_external_release_routing(self):
        model = Model()
        gen = self.construct(model)
        original = gen.cache
        self.assertEqual(len(model.allocations), 1)
        self.assertEqual(gen.page_table.tolist(), [list(range(8)), list(range(8, 16))])
        external = model.external()
        gen.bind_serving_cache(external, page_table=Table([range(8), range(10, 18)]))
        self.assertIs(gen.cache, original)  # Binding never frees standalone allocations.
        gen.release_request(0)
        self.assertIs(self.resets[-1]["cache"], external)
        gen.reset_slots([1])
        self.assertIs(self.resets[-1]["cache"], external)
        gen.reset_slots([1], kv_cache=original)
        self.assertIs(self.resets[-1]["cache"], original)
        self.assertEqual(len(model.allocations), 1)

    def test_reject_invalid_external_metadata_before_sampling_or_reset(self):
        for mutation in (
            lambda cache, table: setattr(cache[1][0].key, "owner", object()),
            lambda cache, table: setattr(cache[1][0].value, "dtype", "bfloat16"),
            lambda cache, table: setattr(cache[0][0].conv, "layout", "ROW_MAJOR"),
            lambda cache, table: setattr(cache[1][0].key, "live", False),
            lambda cache, table: setattr(table, "dtype", "torch.float32"),
            lambda cache, table: table.rows[1].__setitem__(0, 19),
            lambda cache, table: setattr(table, "device", NS(type="cuda")),
        ):
            model = Model()
            cache, table = model.external(), Table([range(8), range(10, 18)])
            mutation(cache, table)
            with self.assertRaises(ValueError):
                self.construct(model, kv_cache=cache, page_table=table)
            self.assertEqual(model.allocations, [])
            self.assertEqual(self.resets, [])
        model = Model()
        with self.assertRaises(ValueError):
            self.construct(model, kv_cache=model.external())
        self.assertEqual(model.allocations, [])

    def test_after_setup_budget_credits_only_actual_retained_target(self):
        budget = DramBudget(
            measured_free_bytes_per_device=(10_000_000,) * 4,
            measurement_artifact="after_setup.json",
            included_categories=tuple(PERSISTENT_CATEGORIES),
            pending_bytes_per_device={},
            reserve_bytes_per_device=1_000_000,
            existing_target_pool_bytes_per_device=(2_000_000,) * 4,
        )
        self.assertEqual(budget.pool_limit().usable_bytes_per_device, (11_000_000,) * 4)
        with self.assertRaises(ValueError):
            plan_cache_allocation(batch=2, context=256, num_blocks=19, budget=budget)

    def test_binding_and_release_respect_pending_trace_closed_lifetimes(self):
        model = Model()
        gen = self.construct(model)
        original = gen.cache_ownership.serving
        external = model.external()
        table = Table([range(8), range(10, 18)])
        for state in ("pending", "trace", "closed"):
            gen.submissions = {1: object()} if state == "pending" else {}
            gen.states = {1: NS(model_trace=7, sample_trace=None)} if state == "trace" else {}
            gen.closed = state == "closed"
            with self.assertRaises(RuntimeError):
                gen.bind_serving_cache(external, page_table=table)
            self.assertIs(gen.cache_ownership.serving, original)
        gen.closed = False
        gen.states = {}
        gen.submissions = {1: object()}
        with self.assertRaises(ValueError):
            gen.release_request(0)
        self.assertEqual(self.resets, [])


if __name__ == "__main__":
    unittest.main()
