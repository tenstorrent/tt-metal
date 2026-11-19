# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host plan-cache and actual attention refresh writer invariants, no TT import."""

import ast
import dataclasses
import functools
import math
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch


class AttentionInputsTests(unittest.TestCase):
    def setUp(self):
        root = Path(os.environ.get("ORNITH_ATTENTION_INPUTS_CANDIDATE_DIR", Path(__file__).resolve().parents[1] / "tt"))
        tree = ast.parse((root / "prefill_ops.py").read_text())
        nodes = [
            n
            for n in tree.body
            if getattr(n, "name", None) in ("AttentionInputs", "make_attention_inputs", "_cached_attention_inputs")
        ]
        self.ns = {
            "dataclass": dataclasses.dataclass,
            "lru_cache": functools.lru_cache,
            "ceil": math.ceil,
            "FULL_PREFILL_BUCKETS": (32, 64),
        }
        exec(compile(ast.Module(body=nodes, type_ignores=[]), "prefill_ops.py", "exec"), self.ns)
        self.make, self.cache = self.ns["make_attention_inputs"], self.ns["_cached_attention_inputs"]
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "NormalizedAttentionPrefill")
        refresh = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "refresh")
        exec(compile(ast.Module(body=[refresh], type_ignores=[]), "prefill_ops.py", "exec"), self.ns)
        self.refresh = self.ns["refresh"]
        self.args = dict(
            physical_bucket=32,
            supported_context=128,
            local_page_count=8,
            start=0,
            valid_count=1,
            page_mapping=[0, 1, 2, 3],
            page_offset=2,
        )
        self.copies, self.fail_on = [], None

        def copy(value, destination):
            self.copies.append(destination)
            destination.copy_(value)
            if len(self.copies) == self.fail_on:
                raise RuntimeError("injected partial transfer")

        fake = SimpleNamespace(
            from_torch=lambda value, **kwargs: value.clone(),
            copy_host_to_device_tensor=copy,
            ROW_MAJOR_LAYOUT="row_major",
            ReplicateTensorToMesh=lambda device: None,
        )
        self.patcher = patch.dict(sys.modules, {"ttnn": fake})
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def make_adapter(self):
        plan = self.make(**self.args)
        state = SimpleNamespace(key=object(), value=object(), page_offset=2, page_count=8, global_pages=False)
        specs = {
            name: ("int", lambda p, name=name: torch.as_tensor(getattr(p, name)).clone())
            for name in ("rope_indices", "fill_pages", "attention_pages", "start")
        }
        return SimpleNamespace(
            state=state,
            _cache_binding=(id(state.key), id(state.value), 2, 8, False),
            bucket=32,
            context=128,
            page_count=8,
            page_offset=2,
            layer=SimpleNamespace(device=None),
            _specs=specs,
            inputs={name: pack(plan) for name, (_, pack) in specs.items()},
            _input_values={name: getattr(plan, name) for name in specs},
            current=None,
        )

    def call(self, adapter, start=0, valid=1, mapping=None):
        return self.refresh(
            adapter, start=start, valid_count=valid, page_mapping=[0, 1, 2, 3] if mapping is None else mapping
        )

    def assert_device_plan(self, adapter, plan):
        for name, tensor in adapter.inputs.items():
            self.assertTrue(torch.equal(tensor, torch.as_tensor(getattr(plan, name))), name)

    def test_pure_plan_identity_immutable_and_mutable_map_snapshot(self):
        mapping = [0, 1, 2, 3]
        first = self.make(**{**self.args, "page_mapping": mapping})
        self.assertIs(first, self.make(**{**self.args, "page_mapping": mapping.copy()}))
        mapping[1] = 7
        changed = self.make(**{**self.args, "page_mapping": mapping})
        self.assertIsNot(first, changed)
        self.assertEqual(first.attention_pages[:4], (2, 3, 4, 5))
        self.assertEqual(changed.attention_pages[:4], (2, 9, 4, 5))
        with self.assertRaises(dataclasses.FrozenInstanceError):
            first.start = 32

    def test_every_scalar_is_validated_or_keyed(self):
        first = self.make(**self.args)
        for change in (
            {"start": 32},
            {"valid_count": 2},
            {"page_offset": 3},
            {"physical_bucket": 64},
            {"local_page_count": 9},
        ):
            self.assertIsNot(first, self.make(**{**self.args, **change}))
        larger = self.make(**{**self.args, "supported_context": 160, "page_mapping": [0, 1, 2, 3, 4]})
        self.assertNotEqual(first.attention_pages, larger.attention_pages)

    def test_invalid_maps_cannot_hit_a_valid_plan(self):
        self.make(**self.args)
        for change in (
            {"page_mapping": [-1, 1, 2, 3]},
            {"page_mapping": [8, 1, 2, 3]},
            {"page_mapping": [0]},
            {"page_offset": -1},
            {"supported_context": 129},
            {"physical_bucket": 48},
            {"start": 1},
            {"valid_count": 33},
        ):
            with self.assertRaises(ValueError):
                self.make(**{**self.args, **change})

    def test_cache_has_eight_entry_bound_and_no_device_owners(self):
        for offset in range(20):
            self.make(**{**self.args, "page_offset": offset})
        self.assertEqual(self.cache.cache_info().currsize, 8)
        self.assertEqual(self.cache.cache_info().maxsize, 8)

    def test_constructor_values_skip_then_position_and_map_changes_copy(self):
        adapter = self.make_adapter()
        plan = self.call(adapter)
        self.assertEqual(len(self.copies), 0)
        self.assertIs(adapter.current, plan)
        plan = self.call(adapter, start=32)
        self.assertEqual(len(self.copies), 3)  # rope, fill, start; same attention map
        self.assert_device_plan(adapter, plan)
        self.call(adapter, start=32)
        self.assertEqual(len(self.copies), 3)
        plan = self.call(adapter, start=32, mapping=[0, 7, 2, 3])
        self.assertEqual(len(self.copies), 5)  # fill and attention maps
        self.assert_device_plan(adapter, plan)
        self.call(adapter, start=32, valid=2, mapping=[0, 7, 2, 3])
        self.assertEqual(len(self.copies), 5)  # valid count alone need not change inputs

    def test_independent_adapters_do_not_share_device_refresh_state(self):
        first, second = self.make_adapter(), self.make_adapter()
        self.call(first, start=32)
        self.call(second, start=32)
        self.assertEqual(len(self.copies), 6)

    def test_partial_failure_blocks_attention_and_old_plan_retry_repairs(self):
        adapter = self.make_adapter()
        self.call(adapter)
        self.fail_on = 2
        with self.assertRaisesRegex(RuntimeError, "injected"):
            self.call(adapter, start=32)
        self.assertIsNone(adapter.current)
        self.assertNotIn("fill_pages", adapter._input_values)
        self.fail_on = None
        plan = self.call(adapter)
        self.assert_device_plan(adapter, plan)
        self.assertIs(adapter.current, plan)

    def test_binding_change_rejected_before_cache_or_device_updates(self):
        adapter = self.make_adapter()
        self.call(adapter)
        before = self.cache.cache_info()
        adapter.state.key = object()
        with self.assertRaisesRegex(ValueError, "Cache binding changed"):
            self.call(adapter)
        self.assertEqual(self.cache.cache_info(), before)
        self.assertEqual(len(self.copies), 0)


if __name__ == "__main__":
    unittest.main()
