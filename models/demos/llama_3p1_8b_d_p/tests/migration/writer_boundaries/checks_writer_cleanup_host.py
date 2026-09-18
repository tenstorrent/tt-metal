# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import ast
import hashlib
import json
import unittest
from pathlib import Path

import writer_boundaries


class PrimaryWriteError(RuntimeError):
    pass


class ReleaseError(RuntimeError):
    pass


class FakeTensor:
    def __init__(self, name, attempts, *, fail=False):
        self.name = name
        self.attempts = attempts
        self.fail = fail

    def deallocate(self, force):
        self.attempts.append((self.name, force))
        if self.fail:
            raise ReleaseError(f"{self.name} release failed")


class WriterCleanupContractTests(unittest.TestCase):
    # A production writer failure is the primary diagnostic, while both temporary
    # tensors still receive exactly one release attempt and all release errors remain recorded.
    def test_primary_writer_error_is_preserved_while_both_releases_are_attempted(self):
        attempts = []
        cleanup_errors = []
        tensors = []
        primary = PrimaryWriteError("writer failed")

        def operation():
            tensors.extend(
                [
                    ("write.k", FakeTensor("k", attempts, fail=True)),
                    ("write.v", FakeTensor("v", attempts, fail=True)),
                ]
            )
            raise primary

        with self.assertRaises(PrimaryWriteError) as caught:
            writer_boundaries.run_with_tensor_cleanup(operation, tensors, cleanup_errors)

        self.assertIs(caught.exception, primary)
        self.assertEqual(attempts, [("v", True), ("k", True)])
        self.assertEqual(
            cleanup_errors,
            [
                "write.v deallocate: ReleaseError: v release failed",
                "write.k deallocate: ReleaseError: k release failed",
            ],
        )

    # The device harness caller must route its real writer and temporary tensors
    # through the tested cleanup helper without importing Torch or TTNN on the host.
    def test_device_write_caller_preserves_writer_error_and_releases_v_then_k(self):
        source = Path("test_writer_boundaries_device.py").read_text()
        module = ast.parse(source)
        write_node = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "_write")
        attempts = []
        cleanup_errors = []
        primary = PrimaryWriteError("device writer failed")

        def tagged_values(phase, kind, slot, layer, positions):
            return kind

        def to_chunk(mesh_device, values):
            return FakeTensor(values, attempts, fail=True)

        def writer(*args, **kwargs):
            raise primary

        namespace = {
            "device_major_positions": lambda start: (start,),
            "_tagged_values": tagged_values,
            "_to_chunk": to_chunk,
            "write_kv_chunk": writer,
            "run_with_tensor_cleanup": writer_boundaries.run_with_tensor_cleanup,
        }
        code = compile(
            ast.Module(body=[write_node], type_ignores=[]),
            str(Path("test_writer_boundaries_device.py")),
            "exec",
        )
        exec(code, namespace)

        with self.assertRaises(PrimaryWriteError) as caught:
            namespace["_write"](
                object(),
                object(),
                phase="write",
                slot=0,
                layer=0,
                start=0,
                end=31,
                cleanup_errors=cleanup_errors,
            )

        self.assertIs(caught.exception, primary)
        self.assertEqual(attempts, [("v", True), ("k", True)])
        self.assertEqual(len(cleanup_errors), 2)

    # Every directly used production allocation/table helper must be hash-pinned,
    # including dependencies whose bytes determine layout, banks, or cache ownership.
    def test_direct_writer_and_table_dependencies_are_pinned(self):
        pins = json.loads(Path("source-pins.json").read_text())
        repository = Path(__file__).resolve().parents[6]
        expected = {
            "models/demos/common/prefill/adapter.py": "8f33b74ebac8cc99ac6c4c42d732316a4e3d1a512559b18801d1f359b2492691",
            "models/demos/common/prefill/runners/migration.py": "3d68639ec2b8e58e53733f3a9194e601a0a0101f18a836d778c4adff20022145",
            "models/demos/llama_3p1_8b_d_p/tt/runners/kv_layout.py": "531e87c02a9e30b8a635fb793c78333181959e49ba3b9b088602dea795e302cf",
        }
        for source, digest in expected.items():
            self.assertEqual(pins.get(source), digest)
            self.assertEqual(hashlib.sha256((repository / source).read_bytes()).hexdigest(), digest)

    # A release-only failure must stop the call after attempting the other tensor,
    # so later writes cannot continue after the harness loses cleanup ownership.
    def test_release_error_fails_successful_write_after_both_release_attempts(self):
        attempts = []
        cleanup_errors = []
        tensors = [
            ("write.k", FakeTensor("k", attempts)),
            ("write.v", FakeTensor("v", attempts, fail=True)),
        ]

        with self.assertRaisesRegex(RuntimeError, "write.v deallocate"):
            writer_boundaries.run_with_tensor_cleanup(lambda: None, tensors, cleanup_errors)

        self.assertEqual(attempts, [("v", True), ("k", True)])
        self.assertEqual(cleanup_errors, ["write.v deallocate: ReleaseError: v release failed"])


if __name__ == "__main__":
    unittest.main()
