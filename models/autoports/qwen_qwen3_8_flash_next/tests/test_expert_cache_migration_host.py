# SPDX-License-Identifier: Apache-2.0
"""Pure migration guards; no live cache mutation or TTNN import."""

import copy
import importlib.util
import unittest
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / "tools/benchmark_migrate_expert_cache.py"
SPEC = importlib.util.spec_from_file_location("expert_cache_migration_host", SOURCE)
TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)

OLD = b"""import math
GLOBAL = 4
class MeshOps:
    def shards(self, weights):
        return weights
class ParallelMoE:
    def __init__(self, ops):
        self.weights = ops.shards([1])
class ParallelGatedDeltaNet:
    def __init__(self):
        self.state = 0
class MultichipDecoder:
    def build(self):
        return self
def _local_recurrence(self):
    return self.state
"""
NEW = OLD.replace(b"self.state = 0", b"self.state = 0\n        self.recurrence_broadcast = False").replace(
    b"return self\n",
    b"obj.attention.recurrence_broadcast = batch_size == 32 and attention_pool_tokens is not None\n        return self\n",
)
NEW += b"""def _broadcast_recurrence_reader(source):
    return source
def _broadcast_recurrence_compute(source):
    return source
_BROADCAST_GDN_READER = 1
_BROADCAST_PREFILL_READER = 2
_BROADCAST_DECODE_COMPUTE = 3
"""


def identities():
    old = {
        "transform_sha256": {"/source.py": TOOL.digest_bytes(OLD), "/cache.py": "cache", "/precision.py": "precision"},
        "native_sha256": {"/native.so": "native"},
        "precision_sha256": "policy",
        "source_seal": {"source": "unchanged"},
    }
    new = copy.deepcopy(old)
    new["transform_sha256"]["/source.py"] = TOOL.digest_bytes(NEW)
    return old, new


class MigrationHostTests(unittest.TestCase):
    def test_exact_gdn_delta_is_allowed(self):
        old, new = identities()
        proof = TOOL.validate_identity_delta(old, new, "/source.py", OLD, NEW)
        self.assertEqual(proof["old_full_ast_sha256"], proof["normalized_new_ast_sha256"])
        self.assertEqual(set(proof["unchanged_producer_ast_sha256"]), {"MeshOps", "ParallelMoE"})

    def test_producer_global_import_and_decoder_changes_are_rejected(self):
        candidates = (
            NEW.replace(b"return weights", b"return weights + [2]"),
            NEW.replace(b"ops.shards([1])", b"ops.shards([2])"),
            NEW.replace(b"GLOBAL = 4", b"GLOBAL = 8"),
            NEW.replace(b"import math", b"import math\nimport os"),
            NEW.replace(b"return self\n", b"return None\n"),
            NEW + b"def unexpected_helper():\n    return 0\n",
        )
        for source in candidates:
            with self.subTest(source=source):
                with self.assertRaisesRegex(ValueError, "Unexpected multichip"):
                    TOOL.validate_source_delta(OLD, source)

    def test_native_policy_cache_precision_and_source_changes_are_rejected(self):
        for field, key in (
            ("native_sha256", "/native.so"),
            ("transform_sha256", "/cache.py"),
            ("transform_sha256", "/precision.py"),
            ("source_seal", "source"),
            ("precision_sha256", None),
        ):
            old, new = identities()
            if key is None:
                new[field] = "changed"
            else:
                new[field][key] = "changed"
            with self.subTest(field=field, key=key):
                with self.assertRaisesRegex(ValueError, "Unexpected cache identity"):
                    TOOL.validate_identity_delta(old, new, "/source.py", OLD, NEW)

    def test_old_source_must_match_pinned_identity(self):
        old, new = identities()
        with self.assertRaisesRegex(ValueError, "Pinned previous source"):
            TOOL.validate_identity_delta(old, new, "/source.py", OLD + b"\n", NEW)

    def test_entry_shape_shards_and_size_are_exact(self):
        shape = [2560, 640]
        entry = {
            "sha256": "a" * 64,
            "bytes": TOOL.BLOB_BYTES,
            "metadata": {
                "shape": shape,
                "padded_shape": shape,
                "shard_shapes": [shape] * 4,
                "dtype": "DataType.BFLOAT4_B",
                "layout": "Layout.TILE",
                "memory_config": TOOL.MEMORY,
            },
        }
        TOOL.validate_entry("layer47/127/up", entry)
        for key in ("layer48/0/up", "layer0/128/up", "layer0/0/down"):
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    TOOL.validate_entry(key, entry)
        altered = copy.deepcopy(entry)
        altered["metadata"]["shard_shapes"] = [shape] * 3
        with self.assertRaisesRegex(ValueError, "shape/format/size"):
            TOOL.validate_entry("layer0/0/up", altered)


if __name__ == "__main__":
    unittest.main()
