"""Bounded capture lifecycle and transition metadata using actual production owner.

CPU stand-ins verify ownership/scheduling, not TT numerical or timing claims.
"""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from models.autoports.qwen_qwen3_8_flash_next.tests.factored_bucket_copy import SOURCE as PROVEN_SOURCE
from models.autoports.qwen_qwen3_8_flash_next.tt.factored_decode import FactoredDecode
from models.autoports.qwen_qwen3_8_flash_next.tt.indexed_rows import SOURCE, metadata_words


class FakeTT:
    bfloat16, uint32, TILE_LAYOUT, ROW_MAJOR_LAYOUT, DRAM_MEMORY_CONFIG = range(5)

    def __init__(self):
        self.allocations, self.executed = [], []

    def zeros(self, shape, **kwargs):
        tensor = torch.zeros(shape)
        self.allocations.append(tuple(shape))
        return tensor

    def from_torch(self, values, **kwargs):
        self.allocations.append(tuple(values.shape))
        return values.clone()

    def execute_trace(self, mesh, trace, **kwargs):
        self.executed.append(trace)


class FactoredDecodeTest(unittest.TestCase):
    def fixture(self, slots):
        tt, uploads, body_calls, records = FakeTT(), [], [], []
        gen = SimpleNamespace(
            max_batch_size=slots,
            mesh=None,
            model=SimpleNamespace(config=SimpleNamespace(hidden_size=2560), layers=[0, 1, 3]),
            caches=list(range(slots)),
            traces={},
            counters={"captures": 0},
            _event=lambda *args, **kwargs: None,
            sample_logits=torch.empty(1, 1, 32, 16),
            _reset_cache=lambda cache: None,
        )

        def refresh(target, value):
            uploads.append(value.clone())
            target.copy_(value)

        gen._refresh = refresh
        owner = FactoredDecode(gen, tt=tt)
        for name in ("clear", "attention", "moe", "terminal"):
            setattr(owner, name, lambda *args, name=name: body_calls.append((name, *args)))

        def record(body, signature=None):
            body()
            records.append(len(records) + 1)
            return records[-1]

        gen._record_trace = record
        return owner, gen, tt, uploads, body_calls

    def test_allocates_configured_slots_and_only_required_buckets(self):
        for count, buckets in ((2, (1, 2)), (3, (1, 2, 4)), (32, (1, 2, 4, 8, 16, 32))):
            owner, gen, tt, _, calls = self.fixture(count)
            self.assertEqual(owner.buckets, buckets)
            self.assertEqual(len(owner.residual), count)
            self.assertEqual(len(tt.allocations), 4 * count + len(buckets) + 1)
            with self.assertRaisesRegex(RuntimeError, "warmed"):
                owner.capture()
            owner.warmup()
            self.assertEqual({call[1] for call in calls if call[0] == "terminal"}, set(range(count)))
            owner.capture()
            self.assertIs(owner.traces, gen.traces)
            self.assertEqual(len(gen.traces), 3 * count + 3 * len(buckets) + count + 1)
            self.assertEqual(gen.counters["captures"], len(gen.traces))

    def test_one_dispatch_plan_no_steady_upload_or_growth(self):
        owner, gen, tt, uploads, _ = self.fixture(3)
        owner.warmup()
        owner.capture()
        allocated, ids = len(tt.allocations), dict(gen.traces)
        for slots in ((2,), (2, 0), (0, 2), (0, 1, 2), (2,)):
            tt.executed.clear()
            owner.replay(slots)
            ordered = tuple(sorted(slots))
            expected = [gen.traces["clear",]]
            bucket = 1 << (len(slots) - 1).bit_length()
            for layer in range(3):
                expected.extend(gen.traces["attention", layer, slot] for slot in ordered)
                expected.append(gen.traces["moe", layer, bucket])
            expected.extend(gen.traces["terminal", slot] for slot in ordered)
            self.assertEqual(tt.executed, expected)
            plan, refreshes = owner._plan, len(uploads)
            owner.replay(tuple(reversed(slots)))
            self.assertIs(owner._plan, plan)
            self.assertEqual(len(uploads), refreshes)
            self.assertEqual(gen.traces, ids)
            self.assertEqual(len(tt.allocations), allocated)
        with self.assertRaises(ValueError):
            owner.replay((3,))
        with self.assertRaises(ValueError):
            owner.replay((1, 1))

    def test_release_invalidates_plan_and_recapture_uses_no_allocation(self):
        owner, gen, tt, _, _ = self.fixture(2)
        owner.warmup()
        owner.capture()
        owner.replay((1,))
        old_ids = set(gen.traces.values())
        before = len(tt.allocations)
        gen.traces.clear()
        owner.invalidate_plan()
        self.assertEqual(owner._plan, ())
        owner.replay((1,))
        self.assertTrue(old_ids.isdisjoint(gen.traces.values()))
        self.assertEqual(len(tt.allocations), before)
        self.assertTrue(set(owner._plan).issubset(gen.traces.values()))

    def test_kernel_matches_hardware_proof_and_metadata_respects_small_owners(self):
        self.assertEqual(SOURCE, PROVEN_SOURCE)
        for owner_size in range(2, 33):
            for count in range(1, owner_size + 1):
                slots = tuple(range(owner_size - count, owner_size))
                indices, state = metadata_words(slots, max_slots=owner_size)
                width = 1 << (count - 1).bit_length()
                self.assertEqual(indices[:count], list(slots))
                self.assertEqual(indices[count:width], [slots[-1]] * (width - count))
                self.assertEqual(state[0], count)
                self.assertTrue(all(0 <= index < owner_size for index in indices[:width]))
        with self.assertRaises(ValueError):
            metadata_words((2,), max_slots=2)

    def test_actual_central_dispatch_and_close_count_native_ids_once(self):
        source = Path(__file__).resolve().parents[1] / "tt/generator.py"
        cls = next(
            n for n in ast.parse(source.read_text()).body if isinstance(n, ast.ClassDef) and n.name == "QwenGenerator"
        )
        names = ("_replay_decode", "close")
        nodes = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
        released, executed, invalidated = [], [], []
        namespace = {
            "ttnn": SimpleNamespace(
                release_trace=lambda mesh, trace: released.append(trace),
                execute_trace=lambda mesh, trace, **kw: executed.append(trace),
            )
        }
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
        owner, gen, _, _, _ = self.fixture(2)
        owner.warmup()
        owner.capture()
        gen.factored_decode = owner
        gen.prefill_checkpoints = SimpleNamespace(invalidate=lambda slots: invalidated.append(tuple(slots)))
        gen.sampler = SimpleNamespace(reset_trace=lambda: None)
        gen.slot_mover = SimpleNamespace(copy_replays={(0, 1): lambda: None})
        gen.prefill_traces, gen.sampling_traces = {4: 2001}, {False: 2002}
        gen.counters.update(model_replays=0, releases=0)
        namespace["_replay_decode"](gen, (1,))
        self.assertEqual(invalidated, [(1,)])
        self.assertEqual(gen.counters["model_replays"], 1)
        ids = set(gen.traces.values()) | {2001, 2002}
        namespace["close"](gen)
        namespace["close"](gen)
        self.assertEqual(set(released), ids)
        self.assertEqual(len(released), len(ids))
        self.assertEqual(gen.counters["releases"], len(ids))
        self.assertEqual(owner._plan, ())
        self.assertEqual(gen.slot_mover.copy_replays, {})
        gen.factored_decode = None
        gen._capture = lambda slots: 987
        namespace["_replay_decode"](gen, (0,))
        self.assertEqual(executed, [987])


if __name__ == "__main__":
    unittest.main()
