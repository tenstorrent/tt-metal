# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU execution of actual prefill refresh methods with owning fake buffers."""

import ast
import os
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

SOURCE = Path(os.environ.get("ORNITH_PREFILL_REFRESH_CANDIDATE_DIR", Path(__file__).resolve().parents[1] / "tt"))


def methods(filename, classname, names):
    tree = ast.parse((SOURCE / filename).read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == classname)
    selected = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in selected} != set(names):
        raise AssertionError("Requested refresh implementation is missing")
    return selected


class FakeTTNN:
    float32, uint32, bfloat16 = "float32", "uint32", "bfloat16"
    TILE_LAYOUT, ROW_MAJOR_LAYOUT = "tile", "row_major"

    def __init__(self):
        self.copies = 0
        self.embeddings = 0
        self.fail_on_copy = None

    def copy_host_to_device_tensor(self, value, destination):
        self.copies += 1
        # Simulate a failed transfer after it changed the destination.
        destination.copy_(value)
        if self.copies == self.fail_on_copy:
            raise RuntimeError("injected transfer failure")

    def embedding(self, indices, source, **kwargs):
        self.embeddings += 1
        return source[indices.long()]

    reshape = staticmethod(torch.reshape)


class PrefillRefreshTests(unittest.TestCase):
    def setUp(self):
        self.tt = FakeTTNN()
        namespace = {
            "torch": torch,
            "ttnn": self.tt,
            "LINEAR_BUCKETS": (32, 64),
            "select_bucket": lambda valid: 32 if valid <= 32 else 64,
        }
        nodes = methods(
            "prefill_ops.py", "NormalizedLinearPrefill", ("__init__", "history_indices", "refresh", "_refresh_profile")
        )
        cls = ast.ClassDef(name="Profile", bases=[], keywords=[], body=nodes, decorator_list=[])
        exec(
            compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), "prefill_ops.py", "exec"),
            namespace,
        )
        self.profile_class = namespace["Profile"]
        self.profile_class.upload = lambda self, value, **kwargs: value.clone()
        child = methods("prefill_ops.py", "RequestProfileLinear", ("refresh_profile",))
        exec(compile(ast.Module(body=child, type_ignores=[]), "prefill_ops.py", "exec"), namespace)
        self.profile_class.refresh_profile = namespace["refresh_profile"]
        self.profile = self.make_profile()
        read = methods("prefill_runtime.py", "PrefillRuntime", ("_read",))
        exec(compile(ast.Module(body=read, type_ignores=[]), "prefill_runtime.py", "exec"), namespace)
        self.read = namespace["_read"]
        self.runtime = SimpleNamespace(
            capacity=128,
            row_indices={height: torch.arange(height, dtype=torch.int32)[None] for height in (32, 64)},
            _row_indices_absolute={32: 0, 64: 0},
            _copy_integer=self.tt.copy_host_to_device_tensor,
        )

    def make_profile(self):
        return self.profile_class(SimpleNamespace(kind="linear_attention", device=None, chunk_size=64))

    def assert_profile(self, profile, bucket, valid):
        buffers = profile.profiles[bucket]
        mask = (torch.arange(bucket) < valid).float()[None, :, None].expand(1, bucket, 8)
        self.assertTrue(torch.equal(buffers["g_mask"], mask))
        self.assertTrue(torch.equal(buffers["beta_mask"], mask))
        self.assertTrue(torch.equal(buffers["history_indices"], profile.history_indices(valid)))
        self.assertEqual(buffers["valid_count"], valid)

    def test_constructor_and_warmup_inputs_are_already_current(self):
        for height in (32, 64):
            self.profile.refresh_profile(valid_count=height, bucket=height)
            self.assert_profile(self.profile, height, height)
        self.assertEqual(self.tt.copies, 0)

    def test_changed_unchanged_and_legacy_writer_share_actual_content(self):
        self.profile.refresh_profile(valid_count=17, bucket=32)
        self.assert_profile(self.profile, 32, 17)
        self.assertEqual(self.tt.copies, 3)
        self.assertEqual(self.profile.refresh(17), 32)
        self.assertEqual(self.tt.copies, 3)
        self.profile.refresh(18)
        self.profile.refresh_profile(valid_count=18, bucket=32)
        self.assert_profile(self.profile, 32, 18)
        self.assertEqual(self.tt.copies, 6)
        self.profile.refresh_profile(valid_count=17, bucket=32)
        self.assertEqual(self.tt.copies, 9)

    def test_profiles_layers_and_slots_use_the_right_persistent_owner(self):
        other_layer = self.make_profile()
        self.profile.refresh_profile(valid_count=17, bucket=32)
        self.profile.refresh_profile(valid_count=17, bucket=64)
        other_layer.refresh_profile(valid_count=17, bucket=32)
        self.assertEqual(self.tt.copies, 9)
        # Request/slot reset changes model state, not these shared immutable inputs.
        self.profile.refresh_profile(valid_count=17, bucket=32)
        self.assertEqual(self.tt.copies, 9)
        self.assert_profile(other_layer, 32, 17)
        self.assert_profile(self.profile, 64, 17)

    def test_partial_failed_transfer_invalidates_old_value(self):
        self.profile.refresh_profile(valid_count=17, bucket=32)
        self.tt.fail_on_copy = self.tt.copies + 2
        with self.assertRaisesRegex(RuntimeError, "injected"):
            self.profile.refresh_profile(valid_count=18, bucket=32)
        self.assertIsNone(self.profile.profiles[32]["valid_count"])
        before = self.tt.copies
        self.tt.fail_on_copy = None
        self.profile.refresh_profile(valid_count=17, bucket=32)
        self.assertEqual(self.tt.copies, before + 3)
        self.assert_profile(self.profile, 32, 17)

    def test_invalid_profile_does_not_mutate_buffers(self):
        for valid, bucket in ((0, 32), (33, 32), (1, 48)):
            with self.assertRaises(ValueError):
                self.profile.refresh_profile(valid_count=valid, bucket=bucket)
        self.assertEqual(self.tt.copies, 0)
        self.assert_profile(self.profile, 32, 32)

    def test_row_indices_change_only_with_absolute_and_preserve_source_reads(self):
        source = torch.arange(128).float()[:, None].expand(128, 4096)
        for absolute, height in ((0, 32), (32, 32), (32, 32), (0, 64), (64, 32), (32, 32)):
            got = self.read(self.runtime, source, absolute, height)
            self.assertTrue(torch.equal(got.reshape(height, 4096), source[absolute : absolute + height]))
        self.assertEqual(self.tt.copies, 3)
        got = self.read(self.runtime, source + 1000, 32, 32)
        self.assertTrue(torch.equal(got.reshape(32, 4096), source[32:64] + 1000))
        self.assertEqual(self.tt.copies, 3)
        self.assertEqual(self.tt.embeddings, 7)

    def test_failed_index_copy_cannot_reuse_old_cached_address(self):
        source = torch.arange(128).float()[:, None].expand(128, 4096)
        self.tt.fail_on_copy = 1
        with self.assertRaisesRegex(RuntimeError, "injected"):
            self.read(self.runtime, source, 32, 32)
        self.assertNotIn(32, self.runtime._row_indices_absolute)
        self.tt.fail_on_copy = None
        got = self.read(self.runtime, source, 0, 32)
        self.assertEqual(self.tt.copies, 2)
        self.assertTrue(torch.equal(got.reshape(32, 4096), source[:32]))

    def test_invalid_read_never_copies_or_embeds(self):
        for absolute, height in ((-1, 32), (100, 32)):
            with self.assertRaises(ValueError):
                self.read(self.runtime, torch.empty(0), absolute, height)
        self.assertEqual((self.tt.copies, self.tt.embeddings), (0, 0))


if __name__ == "__main__":
    unittest.main()
