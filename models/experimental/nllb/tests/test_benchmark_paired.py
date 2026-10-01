# SPDX-FileCopyrightText: © 2026 Abror Shopulatov

# SPDX-License-Identifier: Apache-2.0

"""CPU-only benchmark validation and real existing-loader routing tests."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch

from models.experimental.nllb.benchmarks import benchmark_paired as benchmark
from models.experimental.nllb.benchmarks.benchmark_paired import canonical, checkpoint_identity
from models.experimental.nllb.tt.nllb_validation import load_checkpoint


class OutputTests(unittest.TestCase):
    config = dict(
        vocab_size=32, decoder_start_token_id=2, eos_token_id=2, pad_token_id=1, bos_token_id=0, unk_token_id=3
    )

    def check(self, value, **kw):
        args = dict(config=self.config, batch=1, target_id=7, max_new_tokens=4)
        args.update(kw)
        return canonical(value, **args)

    def test_valid_eos_as_start_and_padding(self):
        rows = np.array([[2, 7, 2, 1, 1], [2, 7, 9, 10, 2]])
        self.assertEqual(self.check(rows, batch=2), [[2, 7, 2], [2, 7, 9, 10, 2]])
        np.testing.assert_array_equal(rows, [[2, 7, 2, 1, 1], [2, 7, 9, 10, 2]])
        self.assertEqual(self.check(np.array([[2, 7]]), max_new_tokens=1), [[2, 7]])
        self.assertEqual(self.check(np.array([[2, 7, 1, 9, 10]])), [[2, 7, 1, 9, 10]])

    def test_declared_ids_not_hardcoded(self):
        cfg = dict(vocab_size=32, decoder_start_token_id=4, eos_token_id=5, pad_token_id=6)
        self.assertEqual(self.check(np.array([[4, 7, 5, 6]]), config=cfg), [[4, 7, 5]])
        cfg["decoder_start_token_id"] = 5
        self.assertEqual(self.check(np.array([[5, 7, 5, 6]]), config=cfg), [[5, 7, 5]])

    def test_wrong_rank_and_container(self):
        for value in ([2, 7, 2], np.array(2), np.array([2, 7, 2]), np.zeros((1, 1, 3), dtype=int)):
            with self.subTest(value=repr(value)), self.assertRaisesRegex(ValueError, "rank-two"):
                self.check(value)

    def test_wrong_batch(self):
        for count in (0, 2, 5):
            with self.subTest(count=count), self.assertRaisesRegex(ValueError, "batch"):
                self.check(np.full((count, 3), 2))

    def test_wrong_length(self):
        for width in (0, 1, 6):
            with self.subTest(width=width), self.assertRaisesRegex(ValueError, "length"):
                self.check(np.full((1, width), 2))

    def test_noninteger_tokens(self):
        for dtype in (float, bool, object, str, complex):
            with self.subTest(dtype=dtype), self.assertRaisesRegex(ValueError, "integer"):
                self.check(np.array([[2, 7, 2]], dtype=dtype))

    def test_invalid_vocabulary_including_hidden_tail(self):
        for value in (-1, 32, 2**63):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "vocabulary"):
                self.check(np.array([[2, 7, 2, value]], dtype=np.uint64 if value == 2**63 else np.int64))

    def test_wrong_start(self):
        with self.assertRaisesRegex(ValueError, "decoder start"):
            self.check(np.array([[0, 7, 2]]))

    def test_wrong_target_and_early_eos(self):
        for target in (2, 8, 1):
            with self.subTest(target=target), self.assertRaisesRegex(ValueError, "forced target"):
                self.check(np.array([[2, target, 2]]))

    def test_nonpad_after_eos_including_repeated_eos(self):
        for tail in ([9], [2], [1, 9], [1, 2]):
            with self.subTest(tail=tail), self.assertRaisesRegex(ValueError, "non-PAD"):
                self.check(np.array([[2, 7, 2] + tail]))

    def test_early_end_without_eos(self):
        for row in ([2, 7], [2, 7, 9], [2, 7, 1, 1]):
            with self.subTest(row=row), self.assertRaisesRegex(ValueError, "without EOS"):
                self.check(np.array([row]))

    def test_reserved_target_policy(self):
        with self.assertRaisesRegex(ValueError, "reserved"):
            self.check(np.array([[2, 2, 2]]), target_id=2)


class RoutingTests(unittest.TestCase):
    def test_single_file_and_directory_use_existing_loader(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            weight = root / "pytorch_model.bin"
            torch.save({"a": torch.tensor([1.0])}, weight)
            config = root / "config.json"
            config.write_text('{"vocab_size":32}')
            for path in (weight, root):
                receipt = checkpoint_identity(path, config)
                self.assertEqual(set(receipt["consumed_checkpoint_files"]), {str(weight)})
                self.assertEqual(
                    receipt["configuration_files"][str(config)]["sha256"],
                    hashlib.sha256(config.read_bytes()).hexdigest(),
                )
                self.assertEqual(load_checkpoint(receipt["loader_path"])["a"].item(), 1.0)

    def test_shards_preferred_and_only_consumed_files_hashed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            torch.save({"ignored": torch.tensor([0.0])}, root / "pytorch_model.bin")
            torch.save({"a": torch.tensor([1.0]), "b": torch.tensor([2.0])}, root / "part1.bin")
            torch.save({"c": torch.tensor([3.0])}, root / "part2.bin")
            index = root / "pytorch_model.bin.index.json"
            index.write_text(json.dumps({"weight_map": {"a": "part1.bin", "b": "part1.bin", "c": "part2.bin"}}))
            config = root / "config.json"
            config.write_text("{}")
            receipt = checkpoint_identity(root, config)
            self.assertEqual(receipt["layout"], "sharded")
            self.assertEqual(
                set(receipt["consumed_checkpoint_files"]),
                {str(index), str((root / "part1.bin").resolve()), str((root / "part2.bin").resolve())},
            )
            with patch.object(torch, "load", wraps=torch.load) as calls:
                weights = load_checkpoint(receipt["loader_path"])
            self.assertEqual(set(weights), {"a", "b", "c"})
            self.assertEqual(calls.call_count, 2)
            for call in calls.call_args_list:
                self.assertEqual(call.kwargs, dict(map_location="cpu", weights_only=True))
            for path, identity in receipt["consumed_checkpoint_files"].items():
                self.assertEqual(identity["sha256"], hashlib.sha256(Path(path).read_bytes()).hexdigest())
            (root / "part2.bin").write_bytes(b"changed")
            self.assertNotEqual(receipt, checkpoint_identity(root, config))

    def test_bad_index_and_missing_shard(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / "config.json"
            config.write_text("{}")
            index = root / "pytorch_model.bin.index.json"
            for mapping in ({}, {"a": "../escape.bin"}, {"a": ""}, {"a": 4}):
                index.write_text(json.dumps({"weight_map": mapping}))
                with self.subTest(mapping=mapping), self.assertRaises(ValueError):
                    checkpoint_identity(root, config)
            index.write_text('{"weight_map":{"a":"x","a":"y"}}')
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                checkpoint_identity(root, config)
            index.write_text('{"weight_map":{"a":"missing.bin"}}')
            with self.assertRaises(FileNotFoundError):
                checkpoint_identity(root, config)


class SourceIdentityTests(unittest.TestCase):
    """Runtime dependencies must participate in the report's before/after comparison."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        original = Path(benchmark.__file__).resolve().parents[1]
        # Explicit inventory makes omission of either runtime dependency fail
        # even when the implementation's SOURCE_FILES list regresses.
        self.required = (
            "__init__.py",
            "benchmarks/benchmark_paired.py",
            "tt/backend.py",
            "tt/nllb_validation.py",
            "tt/generation_projection.txt",
            "tt/trace_decode.py",
            "tt/runtime_setup.py",
        )
        for name in self.required:
            (self.root / name).parent.mkdir(parents=True, exist_ok=True)
            (self.root / name).write_bytes((original / name).read_bytes())
        override = patch.object(benchmark, "PACKAGE", self.root)
        override.start()
        self.addCleanup(override.stop)

    def test_complete_stable_runtime_identity(self):
        before = benchmark.source_identity()
        self.assertEqual(set(before["files"]), set(self.required))
        self.assertEqual(before, benchmark.source_identity())
        for name in ("tt/trace_decode.py", "tt/runtime_setup.py"):
            self.assertEqual(before["files"][name], hashlib.sha256((self.root / name).read_bytes()).hexdigest())

    def check_runtime_mutation(self, name):
        before = benchmark.source_identity()
        path = self.root / name
        original = path.read_bytes()
        self.assertTrue(original.endswith(b"\n"))
        # Equal-size change: identity must depend on bytes, not file size.
        path.write_bytes(original[:-1] + b" ")
        after = benchmark.source_identity()
        self.assertEqual(path.stat().st_size, len(original))
        self.assertNotEqual(before["sha256"], after["sha256"])
        self.assertFalse(after == before, "benchmark source_unchanged comparison must reject runtime drift")
        self.assertEqual({key for key in before["files"] if before["files"][key] != after["files"][key]}, {name})
        path.write_bytes(original)
        self.assertEqual(before, benchmark.source_identity())

    def test_trace_mutation_changes_report_identity(self):
        self.check_runtime_mutation("tt/trace_decode.py")

    def test_runtime_setup_mutation_changes_report_identity(self):
        self.check_runtime_mutation("tt/runtime_setup.py")

    def test_missing_runtime_dependency_fails_closed(self):
        for name in ("tt/trace_decode.py", "tt/runtime_setup.py"):
            with self.subTest(name=name):
                path = self.root / name
                original = path.read_bytes()
                path.unlink()
                try:
                    with self.assertRaises(FileNotFoundError):
                        benchmark.source_identity()
                finally:
                    path.write_bytes(original)


if __name__ == "__main__":
    unittest.main(verbosity=2)
