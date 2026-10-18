# SPDX-License-Identifier: Apache-2.0
"""CPU-only failure guards for the single qualified native rebuild."""

import copy
import unittest
from unittest.mock import patch

from models.autoports.qwen_qwen3_8_flash_next.tools import benchmark_migrate_topk_native_cache as guard


class TopKNativeMigrationTests(unittest.TestCase):
    def objects(self):
        old = {f"unchanged/{i}.o": f"hash{i}" for i in range(1517)}
        old.update({key: hashes[0] for key, hashes in guard.CHANGED_OBJECTS.items()})
        new = dict(old)
        new.update({key: hashes[1] for key, hashes in guard.CHANGED_OBJECTS.items()})
        before = dict(objects=old, build_inputs={"config": "same"}, compiler_entries=["compiler"])
        after = dict(objects=new, build_inputs={"config": "same"}, compiler_entries=["compiler"], build_exit_code=0)
        return before, after

    def test_exact_object_delta(self):
        before, after = self.objects()
        self.assertEqual(guard.validate_object_delta(before, after), guard.CHANGED_OBJECTS)

    def test_rejects_any_unqualified_build_dependency_change(self):
        mutations = [
            lambda d: d["objects"].update({"unchanged/10.o": "other"}),
            lambda d: d["objects"].pop("unchanged/10.o"),
            lambda d: d["objects"].update({"new.o": "new"}),
            lambda d: d["build_inputs"].update(config="other"),
            lambda d: d["compiler_entries"].append("other"),
            lambda d: d.update(build_exit_code=1),
            lambda d: d["objects"].update({next(iter(guard.CHANGED_OBJECTS)): "unqualified"}),
        ]
        for index, mutation in enumerate(mutations):
            before, after = self.objects()
            mutation(after)
            with self.subTest(mutation=index), self.assertRaises(ValueError):
                guard.validate_object_delta(before, after)

    def test_identity_exception_changes_only_exact_native_values(self):
        old = dict(native_sha256={"old": "old"}, transform_sha256={"producer": "same"}, source_seal={"data": "same"})
        new = copy.deepcopy(old)
        new["native_sha256"] = guard.NATIVE_NEW
        guard.validate_identity_delta(old, new)
        for key, value in (("source_seal", {}), ("transform_sha256", {}), ("native_sha256", {}), ("new", 1)):
            changed = dict(new, **{key: value})
            with self.subTest(key=key), self.assertRaises(ValueError):
                guard.validate_identity_delta(old, changed)

    def compatibility(self):
        entries = {
            key: dict(sha256=f"blob{i}", metadata={"shape": [640, 2560]}) for i, key in enumerate(guard.SAMPLE_KEYS)
        }
        proof = dict(
            status="passed",
            old_identity=guard.OLD_IDENTITY,
            old_index_sha256=guard.OLD_INDEX_SHA256,
            build_attestation_sha256="build",
            native_sha256=guard.NATIVE_NEW,
            probe_source_sha256="probe",
            sample_keys=guard.SAMPLE_KEYS,
            cache_written_bytes=0,
            identity=dict(native_sha256=guard.NATIVE_NEW, transform_sha256={"producer": "same"}),
            comparisons=[
                dict(
                    key=key,
                    old_sha256=value["sha256"],
                    fresh_sha256=value["sha256"],
                    metadata=value["metadata"],
                    logical_equal_per_shard=[True] * 4,
                    fresh_path=value["sha256"],
                )
                for key, value in entries.items()
            ],
        )
        return proof, entries

    def fake_sha(self, path):
        return "probe" if path == guard.PROBE_SOURCE else str(path)

    def test_fresh_sample_requires_exact_payload_and_all_shards(self):
        proof, entries = self.compatibility()
        with patch.object(guard, "sha", side_effect=self.fake_sha):
            guard.validate_compatibility(
                proof, "build", entries, dict(native_sha256={}, transform_sha256={"producer": "same"})
            )
        self.assertEqual(len(guard.SAMPLE_KEYS), 12)
        self.assertEqual(len(set(guard.SAMPLE_KEYS)), 12)

    def test_missing_duplicate_wrong_build_or_inexact_sample_rejected(self):
        mutations = [
            lambda p: p["comparisons"].pop(),
            lambda p: p["comparisons"].append(copy.deepcopy(p["comparisons"][0])),
            lambda p: p["comparisons"][0].update(fresh_sha256="different"),
            lambda p: p["comparisons"][0].update(fresh_path="changed-file"),
            lambda p: p["comparisons"][0].update(logical_equal_per_shard=[True, True, False, True]),
            lambda p: p["comparisons"][0].update(metadata={}),
            lambda p: p.update(status="incomplete"),
            lambda p: p.update(old_index_sha256="different"),
            lambda p: p.update(build_attestation_sha256="different"),
            lambda p: p.update(native_sha256={}),
            lambda p: p["identity"].update(transform_sha256={}),
            lambda p: p.update(cache_written_bytes=1),
            lambda p: p.update(probe_source_sha256="different"),
        ]
        for index, mutation in enumerate(mutations):
            proof, entries = self.compatibility()
            mutation(proof)
            with (
                self.subTest(mutation=index),
                patch.object(guard, "sha", side_effect=self.fake_sha),
                self.assertRaises(ValueError),
            ):
                guard.validate_compatibility(
                    proof, "build", entries, dict(native_sha256={}, transform_sha256={"producer": "same"})
                )


if __name__ == "__main__":
    unittest.main()
