# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Production checkpoint lifecycle with CPU tensors; no TTNN imports."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from models.autoports.qwen_qwen3_8_flash_next.tt.prefill_checkpoint import PrefillCheckpoints, compact_state


def cache(position=0):
    attention = SimpleNamespace(
        raw_history=torch.tensor([4.0]),
        position=torch.tensor([position]),
        page_table=torch.arange(4).reshape(1, 4),
        k_cache=torch.tensor([8.0]),
        v_cache=torch.tensor([9.0]),
        compressed_keys=torch.tensor([10.0]),
    )
    layer = SimpleNamespace(
        conv=torch.tensor([1.0]),
        recurrent=torch.tensor([2.0]),
        ple_conv=torch.tensor([3.0]),
        token_history=None,
        attention=attention,
    )
    return SimpleNamespace(layers=[layer], position=position, shared_kv=False)


def method(name, namespace):
    path = Path(__file__).resolve().parents[1] / "tt/generator.py"
    owner = next(
        n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef) and n.name == "QwenGenerator"
    )
    node = next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


class CheckpointLifecycleCPU(unittest.TestCase):
    def setUp(self):
        self.caches = [cache() for _ in range(3)]
        self.copies = []
        self.snapshots = PrefillCheckpoints(
            self.caches,
            allocate=torch.empty_like,
            copy_tensor=lambda source, target: (self.copies.append((source, target)), target.copy_(source)),
        )

    def test_committed_state_eligibility_and_excluded_storage(self):
        current = self.caches[1]
        current.position = 30
        current.layers[0].attention.position.fill_(30)
        self.snapshots.save(1, current, end=31, token=7)
        current.layers[0].recurrent.fill_(99)
        current.position = 31
        current.layers[0].attention.position.fill_(31)
        self.assertEqual(float(current.layers[0].recurrent), 99)
        self.assertTrue(self.snapshots.can_restore(1, 31, 7, 31))
        for start, token, position in ((31, 8, 31), (29, 7, 29), (31, 7, 32), (0, 7, 0)):
            self.assertFalse(self.snapshots.can_restore(1, start, token, position))
        attention = current.layers[0].attention
        attention.page_table.copy_(torch.tensor([[3, 2, 1, 0]]))
        self.snapshots.restore(1, current)
        self.assertEqual(current.position, 30)
        self.assertEqual(float(current.layers[0].recurrent), 2)
        self.assertEqual(attention.page_table.tolist(), [[3, 2, 1, 0]])
        self.assertEqual(
            (float(attention.k_cache), float(attention.v_cache), float(attention.compressed_keys)), (8, 9, 10)
        )
        self.assertFalse(
            any("cache" in name or "compressed" in name or "page_table" in name for name, _ in compact_state(current))
        )

    def test_remap_moves_request_records_without_device_copy_and_invalidates_only_active(self):
        for slot in range(3):
            self.snapshots.save(slot, self.caches[slot], end=31 + 2 * slot, token=slot)
        records = list(self.snapshots.records)
        self.copies.clear()
        self.snapshots.remap((2, 0, 1))
        self.assertFalse(self.copies)
        self.assertEqual([id(r) for r in self.snapshots.records], [id(records[i]) for i in (2, 0, 1)])
        self.snapshots.invalidate((1,))
        self.assertEqual([r.end for r in self.snapshots.records], [35, None, 33])
        before = list(self.snapshots.records)
        with self.assertRaises(ValueError):
            self.snapshots.remap((0, 0, 2))
        self.assertEqual([id(r) for r in before], [id(r) for r in self.snapshots.records])

    def test_actual_decode_invalidates_active_decode_only(self):
        for slot in range(3):
            self.snapshots.save(slot, self.caches[slot], end=31, token=slot)
        replay = method("_replay_decode", {})
        executed = []
        gen = SimpleNamespace(
            prefill_checkpoints=self.snapshots,
            factored_decode=SimpleNamespace(replay=lambda slots: executed.append(slots)),
            counters=dict(model_replays=0),
        )
        replay(gen, (2,))
        self.assertEqual([r.end for r in self.snapshots.records], [31, 31, None])
        self.assertEqual(executed, [(2,)])
        self.assertEqual(gen.counters["model_replays"], 1)

    def test_actual_reset_invalidates_only_owned_cache(self):
        for slot in range(3):
            self.snapshots.save(slot, self.caches[slot], end=31, token=slot)
        namespace = dict(
            torch=torch,
            ttnn=SimpleNamespace(multiply=lambda tensor, value, output_tensor: output_tensor.copy_(tensor * value)),
        )
        reset = method("_reset_cache", namespace)
        gen = SimpleNamespace(
            prefill_checkpoints=self.snapshots,
            caches=self.caches,
            _check_cache_setup=lambda: None,
            counters=dict(cache_resets=0),
            page_tables={0: object(), 1: object()},
            pending_prefill={1: 31},
            _refresh=lambda target, value: target.copy_(value),
        )
        reset(gen, self.caches[1])
        self.assertEqual([r.end for r in self.snapshots.records], [31, None, 31])
        self.assertEqual(self.caches[1].position, 0)
        self.assertIn(0, gen.page_tables)
        self.assertNotIn(1, gen.page_tables)
        self.assertEqual(gen.pending_prefill, {})

    def test_even_end_save_rejected_and_missing_restore_rejected(self):
        with self.assertRaises(ValueError):
            self.snapshots.save(0, self.caches[0], end=32, token=1)
        with self.assertRaises(ValueError):
            self.snapshots.restore(0, self.caches[0])
        self.assertFalse(self.copies)


if __name__ == "__main__":
    unittest.main()
