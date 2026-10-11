# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU integration of scheduler identity and model lifecycle with real storage."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from test_prefix_serving import MemoryDriver

from models.demos.qwen38_27b_qb2.tt.prefix_backend import Qwen38PrefixBackend
from models.demos.qwen38_27b_qb2.tt.prefix_checkpoint import Layout
from models.demos.qwen38_27b_qb2.tt.prefix_storage import AtomicDirectoryStore


class PrefixBackendTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.config = {"layer_types": ["linear_attention"] * 3 + ["full_attention"]}
        self.config["layer_types"] *= 16
        self.weights = self.root / "weights"
        self.weights.mkdir()
        (self.weights / "config.json").write_text(json.dumps(self.config))
        (self.weights / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"w": "one.safetensors"}}))
        (self.weights / "one.safetensors").write_bytes(b"fake immutable weight artifact")
        AtomicDirectoryStore(self.root / "cache", max_bytes=1000000, create=True)
        self.settings = {
            "root": str(self.root / "cache"),
            "max_bytes": 1000000,
            "namespace": "tenant-one",
            "weights_path": str(self.weights),
            "weights_revision": "trusted-artifact-v1",
            "precision": {"kv_cache_dtype": "bfloat8_b", "recurrent_dtype": "float32", "convolution_dtype": "bfloat16"},
            "capture_interval": 32,
        }

    def backend(self):
        backend = Qwen38PrefixBackend(self.settings, self.config)
        # The lifecycle is geometry-independent; tiny opaque buffers keep the
        # CPU test under a megabyte. Native TP4 bytes are separate hardware gates.
        backend.layout = Layout(("full_attention", "linear_attention"), 1, 32, 8, 16, 4, "cpu-test")
        driver = MemoryDriver()
        adapter = SimpleNamespace(
            _complete_prefix_backend=None,
            generator=SimpleNamespace(
                model=SimpleNamespace(
                    mesh=SimpleNamespace(shape=(1, 4)),
                    layer_indices=list(range(64)),
                    precision=self.settings["precision"],
                    config=SimpleNamespace(to_dict=lambda: dict(self.config)),
                    snapshot=self.weights,
                )
            ),
            cache=SimpleNamespace(batch_size=4, num_pages=16),
        )
        with patch("models.demos.qwen38_27b_qb2.tt.prefix_backend.GeneratorPrefixDriver", return_value=driver):
            backend.bind_model(adapter)
        return backend, driver

    def test_capture_lookup_new_private_restore_growth_remap_and_reuse(self):
        backend, driver = self.backend()
        namespace = backend.namespace("salt-a")
        tokens = tuple(range(65))
        with backend.execution():
            self.assertTrue(
                backend.prepare_request(
                    "a", slot=0, tokens=tokens[:32], pages=(1,), namespace=namespace, start=0, checkpoint=None
                )
            )
            backend.complete_request("a", 32, capture=True)
            backend.release_slot(0)
        checkpoint = backend.lookup(tokens, namespace)
        self.assertEqual(checkpoint["consumed"], 32)
        self.assertIsNone(backend.lookup(tokens, backend.namespace("salt-b")))
        with backend.execution():
            self.assertTrue(
                backend.prepare_request(
                    "b", slot=2, tokens=tokens, pages=(4, 5, 6), namespace=namespace, start=32, checkpoint=checkpoint
                )
            )
            backend.complete_request("b", 65, capture=False)
            backend.remap_slots({2: 0, 0: 2})
            self.assertEqual(backend.requests["b"]["slot"], 0)
            backend.refresh_allocations({"b": (4, 5, 6, 7)})
            backend.release_slot(2)  # prior slot no longer owns b
            self.assertIn("b", backend.requests)
            backend.release_slot(0)
            self.assertFalse(backend.requests)
            self.assertTrue(
                backend.prepare_request(
                    "b", slot=0, tokens=tokens, pages=(4, 5, 6), namespace=namespace, start=0, checkpoint=None
                )
            )

    def test_deleted_checkpoint_stays_cold_until_scheduler_retries(self):
        backend, _ = self.backend()
        namespace = backend.namespace(None)
        tokens = tuple(range(64))
        with backend.execution():
            backend.prepare_request(
                "a", slot=0, tokens=tokens, pages=(1, 2), namespace=namespace, start=0, checkpoint=None
            )
            backend.complete_request("a", 32, capture=True)
            backend.release_slot(0)
        checkpoint = backend.lookup(tokens, namespace)
        backend.store.discard(checkpoint["key"])
        with backend.execution():
            self.assertFalse(
                backend.prepare_request(
                    "b", slot=1, tokens=tokens, pages=(3, 4), namespace=namespace, start=32, checkpoint=checkpoint
                )
            )
            self.assertEqual(backend.coordinator.frontier(backend.requests["b"]["handle"]), 0)
            self.assertTrue(
                backend.prepare_request(
                    "b", slot=1, tokens=tokens, pages=(3, 4), namespace=namespace, start=0, checkpoint=None
                )
            )
            backend.complete_request("b", 64, capture=True)

    def test_corrupt_immutable_entry_is_evicted_and_cold_prefill_republishes(self):
        backend, _ = self.backend()
        namespace, tokens = backend.namespace(None), tuple(range(65))
        with backend.execution():
            backend.prepare_request(
                "a", slot=0, tokens=tokens, pages=(1, 2, 3), namespace=namespace, start=0, checkpoint=None
            )
            backend.complete_request("a", 32, capture=True)
            backend.release_slot(0)
        checkpoint = backend.lookup(tokens, namespace)
        (backend.store.root / (checkpoint["key"] + ".checkpoint")).write_bytes(b"truncated")
        with backend.execution():
            self.assertFalse(
                backend.prepare_request(
                    "b", slot=1, tokens=tokens, pages=(4, 5, 6), namespace=namespace, start=32, checkpoint=checkpoint
                )
            )
            self.assertFalse(backend.store.contains(checkpoint["key"]))
            backend.prepare_request(
                "b", slot=1, tokens=tokens, pages=(4, 5, 6), namespace=namespace, start=0, checkpoint=None
            )
            backend.complete_request("b", 32, capture=True)
            self.assertTrue(backend.store.contains(checkpoint["key"]))

    def test_local_huggingface_blob_symlink_is_supported(self):
        shard = self.weights / "one.safetensors"
        blob = self.root / "weight-blob"
        shard.rename(blob)
        shard.symlink_to(blob)
        backend = Qwen38PrefixBackend(self.settings, self.config)
        self.assertEqual(backend.weight_metadata["files"][shard.name][0], blob.stat().st_size)

    def test_artifact_source_precision_namespace_and_salt_partition_identity(self):
        first = Qwen38PrefixBackend(self.settings, self.config)
        tokens = tuple(range(65))
        key = first.checkpoint(tokens, first.namespace(None), 32)
        same = Qwen38PrefixBackend(self.settings, self.config)
        self.assertEqual(key, same.checkpoint(tokens, same.namespace(None), 32))
        self.assertNotEqual(key, same.checkpoint(tokens, same.namespace("other"), 32))
        self.settings["weights_revision"] = "new-artifact"
        changed = Qwen38PrefixBackend(self.settings, self.config)
        self.assertNotEqual(key, changed.checkpoint(tokens, changed.namespace(None), 32))
        self.settings["precision"]["recurrent_dtype"] = "bfloat16"
        with self.assertRaisesRegex(ValueError, "FP32"):
            Qwen38PrefixBackend(self.settings, self.config)

    def test_changed_worker_artifact_or_checkpoint_identity_fails_closed(self):
        backend, _ = self.backend()
        namespace = backend.namespace(None)
        with backend.execution():
            with self.assertRaisesRegex(ValueError, "identities differ"):
                backend.prepare_request(
                    "a",
                    slot=0,
                    tokens=tuple(range(64)),
                    pages=(1, 2),
                    namespace=namespace,
                    start=32,
                    checkpoint={"consumed": 32, "key": "different"},
                )
            backend.release_slot(0)
            backend.prepare_request(
                "b", slot=1, tokens=tuple(range(64)), pages=(3, 4), namespace=namespace, start=0, checkpoint=None
            )
            with self.assertRaisesRegex(ValueError, "without a lifecycle"):
                backend.prepare_request(
                    "b", slot=0, tokens=tuple(range(64)), pages=(3, 4), namespace=namespace, start=0, checkpoint=None
                )
            with self.assertRaisesRegex(ValueError, "without release"):
                backend.refresh_allocations({})


if __name__ == "__main__":
    unittest.main()
