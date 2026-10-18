# SPDX-License-Identifier: Apache-2.0
"""Cache identity and fail-closed host checks; never imports TTNN."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1] / "tt/expert_weight_cache.py"
SPEC = importlib.util.spec_from_file_location("expert_cache_host", SOURCE)
CACHE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CACHE)


class CacheHostTests(unittest.TestCase):
    def test_exact_layer_prefix_resolution_without_weight_reads(self):
        class Checkpoint:
            path = Path("/local/checkpoint")
            prefix = "model.language_model."

            def __getitem__(self, key):
                raise AssertionError("Identity resolution must not read tensor payloads")

        view = SimpleNamespace(state=Checkpoint(), prefix="layers.13.")
        with patch.object(CACHE, "__package__", "cache_test_package"):
            with patch.dict(
                "sys.modules", {"cache_test_package.checkpoint": SimpleNamespace(CheckpointState=Checkpoint)}
            ):
                self.assertEqual(CACHE.locate_state(view), (Checkpoint.path.resolve(), 13))
                view.prefix = "layers.49."
                with self.assertRaises(ValueError):
                    CACHE.locate_state(view)

    def test_atomic_json_detects_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "index.json"
            CACHE.atomic_json(path, {"entry": 3})
            self.assertEqual(CACHE.checked_json(path), {"entry": 3})
            content = json.loads(path.read_text())
            content["payload"]["entry"] = 4
            path.write_text(json.dumps(content))
            with self.assertRaisesRegex(ValueError, "checksum"):
                CACHE.checked_json(path)

    def test_source_seal_reuses_hash_but_rejects_modified_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / CACHE.REVISION
            checkpoint.mkdir()
            (checkpoint / "config.json").write_text("{}")
            name = "weights.safetensors"
            header = json.dumps({"expert": {"dtype": "BF16", "shape": [8], "data_offsets": [0, 16]}}).encode()
            weights = checkpoint / name
            weights.write_bytes(len(header).to_bytes(8, "little") + header + bytes(16))
            (checkpoint / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "weight_map": {
                            "model.language_model.layers.0.mlp.experts.gate_up_proj": name,
                            "model.language_model.layers.0.mlp.experts.down_proj": name,
                        }
                    }
                )
            )
            cache_root = root / "cache"
            sealed = CACHE.seal_sources(checkpoint, cache_root, "fill")
            original_digest = CACHE.digest_file

            def tracked_digest(path):
                self.assertNotEqual(Path(path), weights, "warm load must not hash full source payload again")
                return original_digest(path)

            with patch.object(CACHE, "digest_file", tracked_digest):
                self.assertEqual(CACHE.seal_sources(checkpoint, cache_root, "read"), sealed)
            weights.write_bytes(weights.read_bytes()[:-1] + b"\x01")
            with self.assertRaises(FileNotFoundError):
                CACHE.seal_sources(checkpoint, cache_root, "read")

    def test_read_miss_never_calls_producer(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = CACHE.ExpertWeightCache.__new__(CACHE.ExpertWeightCache)
            cache.mode, cache.entries, cache.root = "read", {}, Path(directory)
            fake_ttnn = SimpleNamespace()

            def forbidden():
                self.fail("Read-only cache miss invoked weight producer")

            with patch.dict("sys.modules", {"ttnn": fake_ttnn}):
                with self.assertRaises(FileNotFoundError):
                    cache.get("layer0/0/gate", (2560, 640), forbidden, None)

    def test_corrupt_blob_is_rejected_before_tensor_load(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = CACHE.ExpertWeightCache.__new__(CACHE.ExpertWeightCache)
            cache.mode, cache.root = "read", Path(directory)
            (cache.root / "blobs").mkdir()
            digest = "a" * 64
            (cache.root / "blobs" / f"{digest}.tensorbin").write_bytes(b"bad")
            cache.entries = {"layer0/0/gate": {"sha256": digest, "bytes": 3}}
            fake_ttnn = SimpleNamespace(load_tensor=lambda *a, **k: self.fail("Corrupt tensor was loaded"))
            with patch.dict("sys.modules", {"ttnn": fake_ttnn}):
                with self.assertRaisesRegex(ValueError, "checksum"):
                    cache.get("layer0/0/gate", (2560, 640), None, None)


if __name__ == "__main__":
    unittest.main()
