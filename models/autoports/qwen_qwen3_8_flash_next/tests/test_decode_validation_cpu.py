# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Run real decode submission validation without importing TTNN or opening devices.

python_env/bin/python -m unittest models.autoports.qwen_qwen3_8_flash_next.tests.test_decode_validation_cpu -v
"""

import ast
import unittest
from dataclasses import dataclass, fields, replace
from pathlib import Path
from types import SimpleNamespace
from typing import List

import torch

from models.autoports.qwen_qwen3_8_flash_next.tt.cache_pool import SharedKVPool
from models.autoports.qwen_qwen3_8_flash_next.tt.host_sampling import HostSamplingParams
from models.autoports.qwen_qwen3_8_flash_next.tt.slot_state import remap_slot_mapping, validate_remap


class DecodeValidationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        common = root.parents[1] / "common/sampling/generator.py"
        from models.autoports.qwen_qwen3_8_flash_next.tt.pending_prefill import plan_pending_prefill

        namespace = dict(
            plan_pending_prefill=plan_pending_prefill,
            dataclass=dataclass,
            fields=fields,
            replace=replace,
            List=List,
            torch=torch,
            SimpleNamespace=SimpleNamespace,
            HostSamplingParams=HostSamplingParams,
            SharedKVPool=SharedKVPool,
            clamp=lambda value, lo, hi: min(max(value, lo), hi),
            validate_remap=validate_remap,
            remap_slot_mapping=remap_slot_mapping,
            remap_token_buffer=lambda tokens, remap: tokens.copy_(
                tokens[..., list(remap) + list(range(len(remap), 32))]
            ),
        )
        nodes = [
            n
            for n in ast.parse(common.read_text()).body
            if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in ("SamplingParams", "format_sampling_params")
        ]
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(common), "exec"), namespace)
        source = root / "tt/generator.py"
        tree = ast.parse(source.read_text())
        owner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "QwenGenerator")
        selected = {
            "set_sampling_params",
            "decode_forward",
            "read_decode_output",
            "process_decode_output_host",
            "_check_cache_setup",
            "_resolve_caches",
            "_table_plan",
        }
        nodes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DecodeSubmission"]
        nodes += [n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name in selected]
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
        cls.gen_type = type("CpuDecode", (), {name: namespace[name] for name in selected})
        cls.params = namespace["SamplingParams"]
        cls.namespace = namespace

    def setUp(self):
        self.events = []
        g = self.gen_type()
        g.max_batch_size, g.capacity = 2, 128
        g.model = SimpleNamespace(config=SimpleNamespace(vocab_size=64))
        g.mesh = object()
        g.caches = [
            SimpleNamespace(
                capacity=128,
                position=9 + i,
                layers=[
                    SimpleNamespace(
                        attention=SimpleNamespace(
                            page_table=torch.arange(4).reshape(1, 4), k_cache=SimpleNamespace(shape=(4, 1, 32, 256))
                        )
                    )
                ],
            )
            for i in range(2)
        ]
        from models.autoports.qwen_qwen3_8_flash_next.tt.prefill_checkpoint import PrefillCheckpoints

        for cache in g.caches:
            cache.layers[0].attention.raw_history = torch.zeros(1)
            cache.layers[0].attention.position = torch.zeros(1, dtype=torch.int32)
        g.prefill_checkpoints = PrefillCheckpoints(
            g.caches, allocate=torch.empty_like, copy_tensor=lambda source, target: target.copy_(source)
        )
        g.pending_prefill = {}
        g.page_tables = {i: torch.arange(4).reshape(1, 4) for i in range(2)}
        g.decode_tokens = torch.arange(32).reshape(1, 1, 1, 32)
        g.saved_tokens = torch.full_like(g.decode_tokens, 99)
        g.sample_logits = torch.zeros(1, 1, 32, 16)
        g.counters = dict(model_replays=0, token_refreshes=0, readbacks=0, synchronizations=0)
        g.prepare = lambda: self.events.append("prepare")
        g.sampler = SimpleNamespace(
            reset_sampling_params=lambda params: self.events.append("params"),
            validate_decode_state_commands=lambda **kw: None,
            apply_slot_remap=lambda remap: self.events.append(("sampler_remap", remap)),
            reset_output_state=lambda **kw: self.events.append(("reset", kw)),
            commit_decode_state_commands=lambda **kw: self.events.append(("commit", kw)),
        )

        def move(remap):
            positions = [c.position for c in g.caches]
            for slot, source in enumerate(remap):
                g.caches[slot].position = positions[source]
            self.events.append(("cache_remap", remap))

        g.slot_mover = SimpleNamespace(apply=move)
        g.seed_state = SimpleNamespace(
            remap=lambda remap: self.events.append(("seed_remap", remap)),
            reset_slots=lambda seeds, positions, slots: self.events.append(("seed_reset", seeds, positions, slots)),
        )

        def bind(tables, caches, slots, positions):
            self.events.append("bind")
            if positions is not None:
                for cache, position in zip(caches, positions, strict=True):
                    cache.position = position

        g._bind = bind
        g._refresh = lambda dest, values: (self.events.append("refresh"), dest.copy_(values))

        def restore(active_slots):
            for slot in range(32):
                if slot not in active_slots:
                    g.decode_tokens[..., slot] = g.saved_tokens[..., slot]

        g._restore_inactive_tokens = restore

        def replay_decode(slots):
            self.events.append("execute")
            g.counters["model_replays"] += 1

        g._replay_decode = replay_decode
        g._sample = lambda slots: self.events.append(("sample", slots))
        self.namespace["ttnn"] = SimpleNamespace(
            copy=lambda src, dst: (self.events.append("copy"), dst.copy_(src)),
            execute_trace=lambda *a, **kw: self.events.append("execute"),
        )
        g.set_sampling_params()
        self.events.clear()
        self.g = g

    def call(self, **changes):
        args = dict(
            tokens=torch.tensor([[3], [4]]),
            start_pos=[9, 10],
            slots=[0, 1],
            kv_cache=self.g.caches,
            page_table=torch.arange(4).repeat(2, 1),
            read_from_device=False,
        )
        args.update(changes)
        return self.g.decode_forward(**args)

    def snapshot(self):
        return (
            self.g.decode_tokens.clone(),
            self.g.saved_tokens.clone(),
            tuple(c.position for c in self.g.caches),
            list(self.g.request_seeds),
            dict(self.g.counters),
            dict(self.g.page_tables),
        )

    def assert_unchanged(self, before):
        after = self.snapshot()
        torch.testing.assert_close(after[0], before[0])
        torch.testing.assert_close(after[1], before[1])
        self.assertEqual(after[2:5], before[2:5])
        for slot in before[5]:
            self.assertIs(after[5][slot], before[5][slot])
        self.assertEqual(self.events, [])

    def test_bad_metadata_rejects_before_remap_and_prepare(self):
        cases = [
            dict(tokens=torch.tensor([3])),
            dict(tokens=torch.ones(2, 2, dtype=torch.long)),
            dict(tokens=torch.tensor([3.0, 4.0])),
            dict(start_pos=[9]),
            dict(start_pos=[9, 1.5]),
            dict(slots=[0]),
            dict(slots=[0, 0]),
            dict(slots=[0, 2]),
            dict(slots=[0.0, 1]),
            dict(kv_cache=[self.g.caches[0]]),
            dict(kv_cache=[self.g.caches[0], object()]),
            dict(page_table=torch.arange(4).reshape(1, 4)),
            dict(page_table=torch.ones(2, 3, dtype=torch.long)),
            dict(page_table=torch.zeros(2, 4)),
            dict(page_table=torch.tensor([[0, 1, 2, 3], [0, 1, 2, 4]])),
            dict(page_table=torch.tensor([[0, 1, 2, 3], [0, 1, 2, -1]])),
            dict(start_pos=[9, 128]),
            dict(tokens=torch.tensor([[3], [64]])),
            dict(reset_batch=False),
            dict(reload_page_table=True),
            dict(reload_inputs=False, reset_sampling_state=True),
            dict(slot_remap=[0, 0]),
            dict(reload_sampling_params=True, sampling_params=self.params(temperature=1.0, top_k=64, top_p=1.0)),
            dict(
                reload_sampling_params=True,
                sampling_params=self.params(seed=[1, 0.5], temperature=[1.0, 1.0], top_k=16, top_p=1.0),
            ),
        ]
        for change in cases:
            with self.subTest(change=change):
                before = self.snapshot()
                args = dict(slot_remap=[1, 0])
                args.update(change)
                with self.assertRaises((ValueError, TypeError)):
                    self.call(**args)
                self.assert_unchanged(before)

    def test_pending_sampler_rejection_precedes_host_and_device_remap(self):
        def reject(**kwargs):
            raise ValueError("pending authoritative sampling reload")

        self.g.sampler.validate_decode_state_commands = reject
        before = self.snapshot()
        with self.assertRaises(ValueError):
            self.call(slot_remap=[1, 0], sample_on_device=True, reload_sampling_params=True, reset_sampling_state=True)
        self.assert_unchanged(before)

    def test_all_inactive_and_zero_rows_are_device_free(self):
        for count in (0, 2):
            for sampled in (False, True):
                for read in (False, True):
                    with self.subTest(count=count, sampled=sampled, read=read):
                        before = self.snapshot()
                        result = self.call(
                            tokens=torch.full((count, 1), -123),
                            start_pos=[-1] * count,
                            slots=list(range(count)),
                            kv_cache=self.g.caches[:count],
                            page_table=None,
                            sample_on_device=sampled,
                            read_from_device=read,
                        )
                        self.assertEqual(result.shape, (count,) if sampled else (count, 64))
                        host, events = self.g.read_decode_output(result, async_read=True)
                        self.assertEqual(events, [])
                        self.assertEqual(
                            self.g.process_decode_output_host(host).shape, (count, 1) if sampled else (count, 1, 64)
                        )
                        self.assert_unchanged(before)

    def test_empty_launch_still_applies_explicit_ownership_remap(self):
        result = self.call(start_pos=[-1, -1], slot_remap=[1, 0])
        self.assertEqual(result.shape, (2, 64))
        self.assertEqual([c.position for c in self.g.caches], [10, 9])
        self.assertEqual(self.g.decode_tokens.flatten()[:2].tolist(), [1, 0])
        self.assertNotIn("execute", self.events)
        self.assertNotIn("bind", self.events)
        self.assertEqual(self.g.counters, dict(model_replays=0, token_refreshes=0, readbacks=0, synchronizations=0))

    def test_authoritative_reload_preserves_inactive_feedback(self):
        before = self.g.decode_tokens.clone()
        result = self.call(start_pos=[11, -1], tokens=torch.tensor([12, -123]), sample_on_device=True)
        self.assertEqual(result.active_rows, (0,))
        self.assertEqual(self.g.decode_tokens.flatten()[0].item(), 12)
        torch.testing.assert_close(self.g.decode_tokens.flatten()[1:], before.flatten()[1:])
        self.assertEqual([c.position for c in self.g.caches], [12, 10])
        self.assertEqual(self.g.counters["token_refreshes"], 1)

    def test_stale_host_values_ignored_and_changed_seed_uses_resident_counter(self):
        before = self.g.decode_tokens.clone()
        self.call(
            tokens=torch.tensor([-999, 10000]),
            start_pos=[10000, 10000],
            reload_inputs=False,
            sample_on_device=True,
            reload_sampling_params=True,
            sampling_params=self.params(seed=[42, 42], temperature=[0.0, 0.0], top_k=1, top_p=1.0),
        )
        torch.testing.assert_close(self.g.decode_tokens, before)
        self.assertIn(("seed_reset", [42, 42], [10, 11], [0, 1]), self.events)
        self.assertEqual(self.g.counters["token_refreshes"], 0)
        self.assertEqual([c.position for c in self.g.caches], [10, 11])

    def test_page_only_keeps_tokens_and_uses_resident_positions(self):
        before = self.g.decode_tokens.clone()
        self.call(
            tokens=torch.tensor([-999, 10000]),
            start_pos=[10000, 10000],
            reload_inputs=False,
            reload_page_table=True,
            page_table=torch.tensor([[3, 2, 1, 0], [3, 2, 1, 0]]),
        )
        torch.testing.assert_close(self.g.decode_tokens, before)
        self.assertIn("bind", self.events)
        self.assertEqual(self.g.counters["token_refreshes"], 0)
        self.assertEqual([c.position for c in self.g.caches], [10, 11])

    def test_remapped_resident_capacity_checked_before_migration(self):
        self.g.caches[1].position = 128
        before = self.snapshot()
        with self.assertRaises(ValueError):
            self.call(start_pos=[9, -1], reload_inputs=False, slot_remap=[1, 0])
        self.assert_unchanged(before)


if __name__ == "__main__":
    unittest.main()
