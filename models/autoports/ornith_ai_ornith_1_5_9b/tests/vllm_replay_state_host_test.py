# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU validation of replay diagnostic indexing; imports no TT runtime."""

import ast
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "tests/vllm_replay_numerics_probe.py"
NAMES = {
    "compact_linear_state",
    "live_physical_pages",
    "logical_kv_prefix",
    "compare_model_tensor",
    "collect_model_state",
    "snapshot_logical_model_state",
    "warm_model_state_reads",
    "summarize_model_states",
}
namespace = {"torch": torch}
exec(  # noqa: S102 -- execute only selected local production definitions, without TT imports.
    compile(
        ast.Module(
            body=[n for n in ast.parse(SOURCE.read_text()).body if isinstance(n, ast.FunctionDef) and n.name in NAMES],
            type_ignores=[],
        ),
        str(SOURCE),
        "exec",
    ),
    namespace,
)

# Compare extraction to the actual decoder reader, not a separately invented layout.
decoder_path = ROOT / "tt/multichip_decoder.py"
tree = ast.parse(decoder_path.read_text())
state_node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DecoderState")
reader = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "read_linear_state")
actual = {
    "dataclass": dataclass,
    "field": field,
    "ttnn": SimpleNamespace(ROW_MAJOR_LAYOUT=None, to_layout=lambda v, layout: v, permute=torch.permute),
}
exec(compile(ast.Module(body=[state_node, reader], type_ignores=[]), str(decoder_path), "exec"), actual)  # noqa: S102
DecoderState = actual["DecoderState"]


class ReplayStateHostTest(unittest.TestCase):
    def test_pooled_and_private_layout_match_actual_decoder_reader(self):
        # Literal allocation shapes in allocate_batch_state: conv[3,1,B,2048],
        # recurrent[B,8,128,128]. Exercise all pooled rows, not just slot zero.
        conv = torch.arange(3 * 3 * 2048).reshape(3, 1, 3, 2048)
        recurrent = torch.arange(3 * 8 * 128 * 128).reshape(3, 8, 128, 128)
        for row in (0, 1, 2):
            state = DecoderState(conv=conv, recurrent=recurrent, state_row=row)
            for name in ("conv", "recurrent"):
                got = namespace["compact_linear_state"](getattr(state, name), name, row)
                expected = actual["read_linear_state"](None, state, name)
                self.assertTrue(torch.equal(got, expected))
        private = DecoderState(conv=torch.zeros(1, 1, 3, 2048), recurrent=torch.zeros(1, 8, 128, 128))
        for name in ("conv", "recurrent"):
            self.assertTrue(
                torch.equal(
                    namespace["compact_linear_state"](getattr(private, name), name, None),
                    actual["read_linear_state"](None, private, name),
                )
            )

    def test_live_mapping_offset_reordering_and_tail_trim(self):
        mapping = torch.tensor([4, 1, 7, 2, 6], dtype=torch.int32)
        for global_pages, offset, expected in ((True, 11, [4, 1, 7, 2, 6]), (False, 11, [15, 12, 18, 13, 17])):
            pages = namespace["live_physical_pages"](
                mapping, 151, global_pages=global_pages, page_offset=offset, physical_page_count=20
            )
            self.assertEqual(pages, expected)
            pool = torch.arange(20 * 32 * 256).reshape(20, 1, 32, 256)
            got = namespace["logical_kv_prefix"]([pool[p : p + 1] for p in pages], 151)
            wanted = pool[expected, 0].reshape(-1, 256)[:151].unsqueeze(0)
            self.assertEqual(tuple(got.shape), (1, 151, 256))
            self.assertTrue(torch.equal(got, wanted))
            self.assertEqual(int(got[0, -1, 0]), int(pool[expected[-1], 0, 22, 0]))

    def test_bad_mapping_is_rejected(self):
        for mapping in ([-1, 1], [1, 1], [1, 20], [1.0, 2.0]):
            with self.subTest(mapping=mapping), self.assertRaises(ValueError):
                namespace["live_physical_pages"](
                    torch.tensor(mapping), 33, global_pages=True, page_offset=0, physical_page_count=20
                )
        with self.assertRaises(ValueError):
            namespace["live_physical_pages"](
                torch.tensor([1]), 33, global_pages=True, page_offset=0, physical_page_count=20
            )

    def snapshots(self):
        names = ("decode", "host_replay", "device_replay")
        return {
            name: {
                "prefix": [7] * 151,
                "logical_end": 151,
                "observed_ends": [151],
                "pages": {"layer3.slot0": [4, 1, 7, 2, 6]},
                "tensors": {
                    "layer0.slot0.conv.rank0": torch.ones(1, 1, 3, 4),
                    "layer0.slot1.recurrent.rank0": torch.zeros(1, 2, 2, 2),
                },
            }
            for name in names
        }

    def test_numerical_decode_drift_is_reported_without_tolerance(self):
        snapshots = self.snapshots()
        snapshots["decode"]["tensors"]["layer0.slot0.conv.rank0"][0, 0, 1, 2] += 0.125
        result = namespace["summarize_model_states"](snapshots)
        self.assertTrue(result["passed"])
        metric = result["comparisons"]["decode_vs_host_replay"]["layer0.slot0.conv.rank0"]
        self.assertFalse(metric["exact"])
        self.assertEqual(metric["max_abs"], 0.125)
        self.assertEqual(metric["worst_index"], [0, 0, 1, 2])

    def test_replay_or_inactive_mismatch_fails(self):
        for key in ("layer0.slot0.conv.rank0", "layer0.slot1.recurrent.rank0"):
            snapshots = self.snapshots()
            snapshots["device_replay"]["tensors"][key] += 1
            result = namespace["summarize_model_states"](snapshots)
            self.assertFalse(result["passed"])
            self.assertFalse(result["checks"]["host_device_replay_exact"])
        snapshots = self.snapshots()
        snapshots["decode"]["tensors"]["layer0.slot1.recurrent.rank0"] += 1
        self.assertFalse(namespace["summarize_model_states"](snapshots)["checks"]["inactive_preserved"])

    def test_off_by_one_nonfinite_shape_and_page_mismatch_fail(self):
        for kind in ("prefix", "end", "nan", "shape", "pages"):
            snapshots = self.snapshots()
            s = snapshots["host_replay"]
            if kind == "prefix":
                s["prefix"] = s["prefix"][:-1]
            if kind == "end":
                s["logical_end"] = 150
            if kind == "nan":
                s["tensors"]["layer0.slot0.conv.rank0"].fill_(float("nan"))
            if kind == "shape":
                s["tensors"]["layer0.slot0.conv.rank0"] = torch.zeros(1)
            if kind == "pages":
                s["pages"] = {"layer3.slot0": [5, 1, 7, 2, 6]}
            self.assertFalse(namespace["summarize_model_states"](snapshots)["passed"], kind)

    def test_collector_reads_only_selected_pages_and_checks_alignment(self):
        conv = torch.arange(3 * 3 * 2048).reshape(3, 1, 3, 2048)
        recurrent = torch.zeros(3, 8, 128, 128)
        key = torch.arange(21 * 32 * 256).reshape(21, 1, 32, 256)
        cache = [
            [DecoderState(conv=conv, recurrent=recurrent, state_row=row) for row in range(3)],
            [DecoderState(key=key, value=key + 1, global_pages=True, page_offset=row * 7) for row in range(3)],
        ]
        mapping = torch.tensor([[4, 1, 7, 2, 6], [10, 12, 13, 14, 15], [11, 19, 20, 8, 9]])
        reads = []

        def read(tensor):
            reads.append(tuple(tensor.shape))
            return tensor

        namespace["ttnn"] = SimpleNamespace(get_device_tensors=lambda tensor: [tensor], to_torch=read)
        gen = SimpleNamespace(
            cache=cache,
            page_table=mapping,
            model=SimpleNamespace(
                layers=[SimpleNamespace(kind="linear_attention"), SimpleNamespace(kind="full_attention")]
            ),
            prefill_runtime=SimpleNamespace(bindings={(id(cache), 0): {"end": 151}}),
        )
        state = SimpleNamespace(positions=[torch.tensor([151])], host_tables=mapping)
        decoded = namespace["collect_model_state"](gen, state, [7] * 151, phase="decode", layer_ids=[0, 3])
        replay = namespace["collect_model_state"](gen, state, [7] * 151, phase="prefill", layer_ids=[0, 3])
        self.assertTrue(all(torch.equal(value, replay["tensors"][name]) for name, value in decoded["tensors"].items()))
        self.assertTrue(
            torch.equal(
                decoded["tensors"]["layer3.slot0.key.rank0"], key[[4, 1, 7, 2, 6], 0].reshape(1, -1, 256)[:, :151]
            )
        )
        self.assertEqual(reads.count((1, 1, 32, 256)), 2 * 2 * (5 + 1 + 1))
        self.assertNotIn(tuple(key.shape), reads)
        state.positions[0] = torch.tensor([150])
        with self.assertRaises(AssertionError):
            namespace["collect_model_state"](gen, state, [7] * 151, phase="decode", layer_ids=[0, 3])

    def test_diagnostic_slice_keys_warmed_before_capture(self):
        programs, late = set(), set()
        captured = False

        class Shard:
            shape = (24576, 1, 32, 256)

            def __init__(self, name, rank):
                self.name, self.rank = name, rank

            def __getitem__(self, index):
                signature = (self.name, self.rank, self.shape, index.start, index.stop)
                if captured and signature not in programs:
                    late.add(signature)
                programs.add(signature)
                return torch.full((1, 1, 32, 256), float(index.start))

        class Pool:
            shape = Shard.shape

            def __init__(self, name):
                self.shards = [Shard(name, rank) for rank in range(4)]

        key, value = Pool("key"), Pool("value")
        cache = [[DecoderState(key=key, value=value, global_pages=True, page_offset=i * 8192) for i in range(3)]]
        mapping = torch.stack([torch.arange(i * 8192, (i + 1) * 8192) for i in range(3)])
        state = SimpleNamespace(model_trace=None, host_tables=mapping, positions=[torch.tensor([151])])
        gen = SimpleNamespace(
            cache=cache,
            page_table=mapping,
            model=SimpleNamespace(layers=[SimpleNamespace(kind="full_attention")]),
            states={0: state},
            device=SimpleNamespace(num_program_cache_entries=lambda: len(programs)),
            prefill_runtime=SimpleNamespace(bindings={(id(cache), 0): {"end": 151}}),
        )
        namespace["ttnn"] = SimpleNamespace(
            get_device_tensors=lambda value: value.shards if isinstance(value, Pool) else [value],
            to_torch=lambda value: value,
        )
        # Negative control: the old post-capture first read creates late keys.
        captured = True
        namespace["collect_model_state"](gen, state, [7] * 151, phase="decode", layer_ids=[3])
        self.assertEqual(len(late), 56)
        programs.clear()
        late.clear()
        captured = False
        result = namespace["warm_model_state_reads"](gen, state, length=151, layer_ids=[3])
        self.assertEqual(
            (result["programs_before"], result["programs_after_first"], result["programs_after_second"]), (0, 56, 56)
        )
        expected = {
            (name, rank, Shard.shape, page, page + 1)
            for name in ("key", "value")
            for rank in range(4)
            for page in (0, 1, 2, 3, 4, 8192, 16384)
        }
        self.assertEqual(programs, expected)
        captured = True
        state.model_trace = object()
        for phase in ("decode", "prefill"):
            namespace["collect_model_state"](gen, state, [7] * 151, phase=phase, layer_ids=[3])
        self.assertFalse(late)
        with self.assertRaisesRegex(RuntimeError, "before capture"):
            namespace["warm_model_state_reads"](gen, state, length=151, layer_ids=[3])
        state.model_trace = None
        gen.page_table = mapping.flip(0)
        with self.assertRaisesRegex(AssertionError, "mappings differ"):
            namespace["warm_model_state_reads"](gen, state, length=151, layer_ids=[3])

    def test_diagnostic_warmup_precedes_capture_and_is_opt_in(self):
        main = next(
            n for n in ast.parse(SOURCE.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "main"
        )
        warm = next(
            n
            for n in ast.walk(main)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "warm_model_state_reads"
        )
        capture = next(
            n
            for n in ast.walk(main)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "capture"
        )
        self.assertLess(warm.lineno, capture.lineno)
        parents = {child: node for node in ast.walk(main) for child in ast.iter_child_nodes(node)}
        node = warm
        while node in parents:
            node = parents[node]
            if isinstance(node, ast.If) and ast.unparse(node.test) == "args.compare_model_state":
                break
        else:
            self.fail("Diagnostic warmup is not opt-in guarded")

    def test_default_has_no_model_state_readbacks(self):
        tree = ast.parse(SOURCE.read_text())
        parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
        calls = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "collect_model_state"
        ]
        self.assertEqual(len(calls), 3)
        for call in calls:
            node = call
            while node in parents:
                node = parents[node]
                if isinstance(node, ast.If) and ast.unparse(node.test) == "args.compare_model_state":
                    break
            else:
                self.fail("State collection is not opt-in guarded")


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
