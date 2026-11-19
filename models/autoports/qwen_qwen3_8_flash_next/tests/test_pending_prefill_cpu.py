# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Real prefill orchestration and fixed-address state-copy regression, no devices."""

import ast
import copy
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from models.autoports.qwen_qwen3_8_flash_next.tests import test_prefill_validation_cpu as prefill_validation
from models.autoports.qwen_qwen3_8_flash_next.tt.cache_pool import SharedKVPool
from models.autoports.qwen_qwen3_8_flash_next.tt.pending_prefill import plan_pending_prefill
from models.autoports.qwen_qwen3_8_flash_next.tt.prefill_checkpoint import PrefillCheckpoints
from models.autoports.qwen_qwen3_8_flash_next.tt.slot_state import SlotStateMover, _state_tensors


class PendingPrefillTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        prefill_validation.PrefillValidationTest.setUpClass.__func__(cls)
        path = Path(__file__).resolve().parents[1] / "tt/generator.py"
        owner = next(
            n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef) and n.name == "QwenGenerator"
        )
        nodes = [
            n
            for n in owner.body
            if isinstance(n, ast.FunctionDef) and n.name in ("_bind", "_reset_cache", "release_request")
        ]
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), cls.namespace)
        cls.real_methods = {n.name: cls.namespace[n.name] for n in nodes}

    def setUp(self):
        prefill_validation.PrefillValidationTest.setUp(self)
        g = self.g
        for name, method in self.real_methods.items():
            setattr(g, name, method.__get__(g, self.gen_type))
        g.shared_pool = SharedKVPool(
            [((16, 1, 32, 256), "bfloat16", 0)],
            allocate_pair=lambda _, spec: tuple(
                SimpleNamespace(shape=spec.shape, dtype="bfloat16", layout="tile") for _ in range(2)
            ),
        )
        g.page_tables = {}
        for i, cache in enumerate(g.caches):
            cache.shared_kv = True
            cache.shared_kv_pool = g.shared_pool
            cache.position = 0
            layer = cache.layers[0]
            for name in ("conv", "recurrent", "ple_conv", "token_history"):
                setattr(
                    layer,
                    name,
                    torch.full(
                        (1, 16) if name == "token_history" else (1,),
                        i + 11,
                        dtype=torch.int32 if name == "token_history" else torch.float32,
                    ),
                )
            layer.attention.compressed_keys = torch.full((1,), i + 21.0)
        g.prefill_cache = copy.deepcopy(g.caches[0])
        g.slot_mover = SlotStateMover(g.caches, g.prefill_cache, copy_tensor=lambda src, dst: dst.copy_(src))
        g.prefill_checkpoints = PrefillCheckpoints(
            g.caches, allocate=torch.empty_like, copy_tensor=lambda src, dst: dst.copy_(src)
        )
        g.counters.update(cache_resets=0, page_table_refreshes=0, position_refreshes=0)
        self.namespace["ttnn"].multiply = lambda tensor, factor, output_tensor: output_tensor.mul_(factor)
        self.namespace["EOS"] = 1
        self.transitions = []
        g._event = lambda kind, **data: self.transitions.append((kind, data))

    def seed(self, slot, position, pages, *, pending=False):
        self.g.caches[slot].position = position
        table = torch.tensor([pages + [0] * (4 - len(pages))], dtype=torch.int32)
        self.g.page_tables[slot] = (table,)
        self.g.caches[slot].layers[0].attention.page_table.copy_(table)
        if pending:
            self.g.pending_prefill[slot] = position
        return table

    def submit(self, table, *, start=120, end=128, slot=0, **kwargs):
        return self.g.prefill_forward(
            tokens=torch.arange(128).remainder(64).reshape(1, -1),
            prompt_lens=[end],
            start_pos=[start],
            slots=[slot],
            kv_cache=None,
            page_table=table,
            sampling_params=self.params(temperature=0.0, top_k=1, top_p=1.0),
            **kwargs,
        )

    def snapshot_state(self):
        return (
            [[(name, id(t), t.clone()) for name, t in _state_tensors(c)] for c in self.g.caches],
            [c.position for c in self.g.caches],
            dict(self.g.pending_prefill),
            copy.deepcopy(self.g.page_tables),
            list(self.events),
            list(self.transitions),
        )

    def assert_state(self, before):
        after = self.snapshot_state()
        for left, right in zip(before[0], after[0]):
            for (name, identity, value), (new_name, new_identity, new_value) in zip(left, right):
                self.assertEqual((name, identity), (new_name, new_identity))
                self.assertTrue(torch.equal(value, new_value), name)
        self.assertEqual(before[1:3], after[1:3])
        self.assertEqual(before[4:], after[4:])
        for slot in before[3]:
            self.assertTrue(torch.equal(before[3][slot][0], after[3][slot][0]))

    def test_released_sibling_moves_all_state_before_final_tail(self):
        g = self.g
        self.seed(0, 158, [1, 2, 3, 4])
        table = self.seed(1, 120, [5, 6, 7, 8], pending=True)
        expected = {name: tensor.clone() for name, tensor in _state_tensors(g.caches[1])}
        identities = [[id(t) for _, t in _state_tensors(c)] for c in g.caches]
        g.release_request(0)
        with self.assertRaisesRegex(ValueError, "Overlapping"):
            g._table_plan(table, [g.caches[0]], (0,), [128])
        original = g._prefill_chunk
        observed = []

        def prefill(n, slot):
            if not observed:
                for name, tensor in _state_tensors(g.prefill_cache):
                    if not name.endswith(".position"):
                        self.assertTrue(torch.equal(tensor, expected[name]), name)
                observed.append(True)
            original(n, slot)

        g._prefill_chunk = prefill
        self.assertEqual(self.submit(table).tolist(), [[63]])
        self.assertEqual([c.position for c in g.caches], [128, 0])
        self.assertEqual(g.pending_prefill, {})
        self.assertEqual(set(g.page_tables), {0})
        self.assertEqual(identities, [[id(t) for _, t in _state_tensors(c)] for c in g.caches])
        self.assertEqual(
            [data for kind, data in self.transitions if kind == "pending_prefill_relocation"],
            [{"moves": [{"source": 1, "destination": 0, "position": 120}]}],
        )
        self.assertFalse(any(isinstance(e, tuple) and e[0] in ("sampler_remap", "seed_remap") for e in self.events))

    def test_invalid_identity_or_sampling_never_mutates(self):
        cases = ("wrong_prefix", "active_source", "occupied_destination", "invalid_page", "invalid_sampling")
        for case in cases:
            with self.subTest(case=case):
                self.setUp()
                table = self.seed(1, 120, [5, 6, 7, 8], pending=case != "active_source")
                kwargs = {}
                if case == "wrong_prefix":
                    table = table.clone()
                    table[0, 2] = 9
                if case == "occupied_destination":
                    self.seed(0, 8, [1])
                if case == "invalid_page":
                    table = table.clone()
                    table[0, 3] = 99
                if case == "invalid_sampling":
                    kwargs["return_all_logits"] = True
                before = self.snapshot_state()
                with self.assertRaises(ValueError):
                    self.submit(table, **kwargs)
                self.assert_state(before)

    def test_odd_checkpoint_moves_before_restore_and_current_table_binding(self):
        g = self.g
        table = self.seed(1, 119, [5, 6, 7, 8], pending=True)
        g.caches[1].layers[0].recurrent.fill_(73)
        g.prefill_checkpoints.save(1, g.caches[1], end=119, token=118 % 64)
        g.caches[1].layers[0].recurrent.fill_(91)
        record = g.prefill_checkpoints.records[1]
        original = g._prefill_chunk
        seen = []

        def prefill(n, slot):
            if not seen:
                self.assertEqual(g.prefill_cache.position, 118)
                self.assertEqual(g.prefill_cache.layers[0].recurrent.item(), 73)
                seen.append(True)
            original(n, slot)

        g._prefill_chunk = prefill
        self.submit(table, start=119)
        self.assertIs(g.prefill_checkpoints.records[0], record)
        self.assertEqual(self.chunks[0][1][0], 118 % 64)
        self.assertEqual([c.position for c in g.caches], [128, 0])

    def test_same_slot_cow_keeps_owner_and_no_movement(self):
        self.seed(1, 120, [5, 6, 7, 8], pending=True)
        self.submit(torch.tensor([[9, 10, 11, 12]], dtype=torch.int32), slot=1)
        self.assertFalse(any(kind == "pending_prefill_relocation" for kind, _ in self.transitions))
        self.assertEqual(self.g.pending_prefill, {})

    def test_prefix_growth_per_layer_and_ambiguity(self):
        a = torch.tensor([[5, 6, 0, 0]], dtype=torch.int32)
        b = torch.tensor([[9, 10, 0, 0]], dtype=torch.int32)
        args = dict(
            tables=(a, b),
            shadows={1: (a.clone(), b.clone())},
            positions=[0, 32],
            pending={1: 32},
            slots=(0,),
            starts=(32,),
        )
        self.assertEqual(plan_pending_prefill(**args), (1, 0))
        changed = b.clone()
        changed[0, 0] = 11
        with self.assertRaises(ValueError):
            plan_pending_prefill(**dict(args, tables=(a, changed)))
        with self.assertRaisesRegex(ValueError, "Ambiguous"):
            plan_pending_prefill(
                **dict(args, positions=[32, 32], pending={0: 32, 1: 32}, shadows={0: (a, b), 1: (a, b)})
            )
        with self.assertRaisesRegex(ValueError, "same pending owner"):
            plan_pending_prefill(**dict(args, tables=(a.repeat(2, 1), b.repeat(2, 1)), slots=(0, 1), starts=(32, 32)))

    def test_new_duplicate_page_and_bad_later_row_reject_before_movement(self):
        table = self.seed(1, 32, [5], pending=True)
        table[0, 1] = 5
        before = self.snapshot_state()
        with self.assertRaisesRegex(ValueError, "Overlapping"):
            self.submit(table, start=32, end=40)
        self.assert_state(before)
        table = self.seed(1, 120, [5, 6, 7, 8], pending=True)
        tokens = torch.zeros(2, 128, dtype=torch.int32)
        tokens[1, 0] = 64
        before = self.snapshot_state()
        with self.assertRaisesRegex(ValueError, "vocabulary"):
            self.g.prefill_forward(
                tokens,
                page_table=table.repeat(2, 1),
                kv_cache=None,
                prompt_lens=[128, 8],
                start_pos=[120, 0],
                slots=[0, 1],
            )
        self.assert_state(before)

    def test_simultaneous_pending_cycle_preserves_unsubmitted_holder(self):
        a = torch.tensor([[5, 0, 0, 0]], dtype=torch.int32)
        b = torch.tensor([[6, 0, 0, 0]], dtype=torch.int32)
        result = plan_pending_prefill(
            tables=(torch.cat([b, a]),),
            shadows={0: (a,), 1: (b,)},
            positions=[32, 32, 77],
            pending={0: 32, 1: 32},
            slots=(0, 1),
            starts=(32, 32),
        )
        self.assertEqual(result, (1, 0, 2))

    def test_pending_lifecycle_and_decode_permutation(self):
        g = self.g
        table = self.seed(1, 112, [5, 6, 7, 8], pending=True)
        self.submit(table, start=112, end=120, slot=1, prefill_sample_mask=[False])
        self.assertEqual(g.pending_prefill, {1: 120})
        # Full accepted decode remap moves an unscheduled partial-prefill owner.
        g._replay_decode = lambda slots: None
        g.decode_forward(
            torch.tensor([[2]]),
            [-1],
            slots=[0],
            kv_cache=None,
            page_table=None,
            slot_remap=(1, 0),
            read_from_device=False,
        )
        self.assertEqual(g.pending_prefill, {0: 120})
        g.release_request(0)
        self.assertEqual(g.pending_prefill, {})
        self.assertNotIn(0, g.page_tables)


if __name__ == "__main__":
    unittest.main()
