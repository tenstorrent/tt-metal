# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only planner checks. Run as a module to bypass device pytest fixtures."""

import math
import subprocess
import sys
import unittest
from itertools import pairwise
from pathlib import Path

from models.autoports.ornith_ai_ornith_1_5_9b.tt.precision import prefill_l1_row_limit, prefill_uses_dram
from models.autoports.ornith_ai_ornith_1_5_9b.tt.prefill_plan import (
    CHUNK_CAPS,
    FULL_PREFILL_BUCKETS,
    LINEAR_BUCKETS,
    MAX_CONTEXT,
    physical_bucket,
    plan_prefill,
    prefix_profiles_compatible,
)


def original_shape_policy(kind, count):
    """Explicit original decoder Python branches; no device execution claim."""
    n = count if kind == "linear_attention" else math.ceil(count / 32) * 32
    special = 512 <= n <= 1024
    return (
        n == 1,
        n >= 512,  # norm output / reduction payload BF8, otherwise BF16
        n > 4096,  # special large-prefill norm path
        n > 8192,  # DRAM projection activation / norm output residency
        special,
        math.ceil(n / (320 if kind == "linear_attention" else 256)) if special else None,
        math.ceil(n / 256) if special else None,
        ("single" if n == 1 else math.ceil(n / 32)) if n <= 64 else ("short128" if n <= 128 else "fused"),
    )


class PrefillPlanTests(unittest.TestCase):
    def test_precision_placement_uses_physical_bucket_and_preserves_capacity(self):
        baseline = prefill_l1_row_limit("bfloat8_b", ["bfloat8_b"] * 4)
        for activation, weights in (("bfloat16", ["bfloat8_b"] * 4), ("bfloat8_b", ["bfloat16", "bfloat8_b"])):
            limit = prefill_l1_row_limit(activation, weights)
            for kind in CHUNK_CAPS:
                for count in (4095, 4096, 4097, 8191, 8192, 8193):
                    bucket = physical_bucket(kind, count)
                    self.assertGreater(bucket, limit)
                    self.assertEqual(bucket > baseline, count > 8192)
                # A tiny suffix can execute a large horizon-derived profile.
                plan = plan_prefill(total_length=4097, start=1, end=2, kind=kind)
                self.assertTrue(all(s.physical_bucket > limit for s in plan.segments))
        for count, uses_dram in ((4064, False), (4065, True), (4095, True), (4096, True)):
            self.assertEqual(prefill_uses_dram(count, 4095), uses_dram)
        self.assertEqual(MAX_CONTEXT, 262144)

    def assert_plan(self, total, start, end, kind):
        plan = plan_prefill(total_length=total, start=start, end=end, kind=kind)
        cursor = start
        cap = CHUNK_CAPS[kind]
        for segment in plan.segments:
            self.assertEqual(segment.logical_start, cursor)
            self.assertGreater(segment.logical_valid, 0)
            self.assertEqual(segment.absolute_aligned_start % 32, 0)
            self.assertLess(segment.initial_overlap, 32)
            self.assertLessEqual(segment.valid_count, segment.physical_bucket)
            self.assertEqual(segment.valid_count, segment.logical_valid + segment.initial_overlap)
            self.assertEqual(segment.canonical_chunk_start, cursor // cap * cap)
            self.assertEqual(segment.canonical_chunk_end, min(total, cursor // cap * cap + cap))
            self.assertLessEqual(segment.logical_end, segment.canonical_chunk_end)
            self.assertGreaterEqual(segment.absolute_aligned_start, segment.canonical_chunk_start)
            if segment.initial_overlap:
                saved = segment.required_cached_boundary
                self.assertIsNotNone(saved)
                self.assertEqual((saved.boundary, saved.tail_end), (segment.absolute_aligned_start, cursor))
                self.assertEqual(saved.requires_conv_recurrent, kind == "linear_attention")
            else:
                self.assertIsNone(segment.required_cached_boundary)
            cursor = segment.logical_end
        self.assertEqual(cursor, end)
        self.assertEqual(sum(s.logical_valid for s in plan.segments), end - start)
        needs_tail = end < total and end % 32 != 0
        saved = [s.save_boundary_before for s in plan.segments if s.save_boundary_before is not None]
        self.assertEqual(len(saved), int(needs_tail))
        if saved:
            self.assertEqual((saved[0].boundary, saved[0].tail_end), (end // 32 * 32, end))
        return plan

    def test_catalogues_preserve_explicit_policy_for_every_canonical_count(self):
        for kind, cap in CHUNK_CAPS.items():
            for count in range(1, cap + 1):
                bucket = physical_bucket(kind, count)
                self.assertEqual(original_shape_policy(kind, count), original_shape_policy(kind, bucket), (kind, count))
        self.assertEqual(physical_bucket("linear_attention", 511), 511)
        self.assertEqual(physical_bucket("full_attention", 511), 512)
        self.assertEqual(physical_bucket("linear_attention", 513), 640)
        self.assertEqual(physical_bucket("full_attention", 513), 768)

    def test_fragment_and_cap_boundaries(self):
        for kind, cap in CHUNK_CAPS.items():
            for total in (1, 31, 33, 511, 513, 16383, 16385, 32769, 262143, MAX_CONTEXT):
                points = sorted(
                    {
                        0,
                        total,
                        *(
                            p
                            for p in (1, 2, 31, 32, 33, 63, 64, 68, 69, 511, 512, cap - 1, cap, cap + 1, total - 1)
                            if 0 < p < total
                        ),
                    }
                )
                for start, end in pairwise(points):
                    self.assert_plan(total, start, end, kind)
                self.assert_plan(total, 0, total, kind)
                for start in points[:-1]:
                    self.assert_plan(total, start, total, kind)

    def test_prefix68_retains_boundary64(self):
        for kind in CHUNK_CAPS:
            prefix = self.assert_plan(513, 0, 68, kind)
            self.assertEqual([(s.logical_start, s.logical_end) for s in prefix.segments], [(0, 64), (64, 68)])
            suffix = self.assert_plan(513, 68, 513, kind)
            self.assertEqual(suffix.segments[0].valid_count, 449)
            self.assertEqual(suffix.segments[0].initial_overlap, 4)
            self.assertEqual(prefix.segments[-1].save_boundary_before, suffix.segments[0].required_cached_boundary)

    def test_same_tile_continuation_and_next_snapshot(self):
        for kind in CHUNK_CAPS:
            segment = self.assert_plan(513, 68, 70, kind).segments[0]
            self.assertEqual(segment.valid_count, 6)
            self.assertEqual(segment.required_cached_boundary.tail_rows, 4)
            self.assertEqual(segment.save_boundary_before.tail_rows, 6)

    def test_every_small_fragment_preserves_token_coverage(self):
        for kind in CHUNK_CAPS:
            for start in range(96):
                for end in range(start + 1, 97):
                    self.assert_plan(97, start, end, kind)

    def test_whole_plan_keeps_original_caps(self):
        for kind, cap in CHUNK_CAPS.items():
            plan = self.assert_plan(MAX_CONTEXT, 0, MAX_CONTEXT, kind)
            self.assertEqual(len(plan.segments), MAX_CONTEXT // cap)
            self.assertTrue(all(s.valid_count == cap and s.physical_bucket == cap for s in plan.segments))

    def test_profile_compatibility_requires_known_matching_profiles(self):
        for kind in CHUNK_CAPS:
            self.assertTrue(
                prefix_profiles_compatible(
                    cached_total_length=513, requested_total_length=600, prefix_end=68, kind=kind
                )
            )
            self.assertFalse(
                prefix_profiles_compatible(
                    cached_total_length=511, requested_total_length=513, prefix_end=68, kind=kind
                )
            )
            self.assertFalse(
                prefix_profiles_compatible(
                    cached_total_length=513, requested_total_length=None, prefix_end=68, kind=kind
                )
            )
            cap = CHUNK_CAPS[kind]
            self.assertTrue(
                prefix_profiles_compatible(
                    cached_total_length=cap + 1, requested_total_length=cap + 513, prefix_end=cap, kind=kind
                )
            )
            self.assertFalse(
                prefix_profiles_compatible(
                    cached_total_length=cap + 1, requested_total_length=cap + 513, prefix_end=cap + 1, kind=kind
                )
            )

    def test_invalid_inputs(self):
        for kwargs in (
            {"total_length": 0, "start": 0, "end": 1},
            {"total_length": MAX_CONTEXT + 1, "start": 0, "end": 1},
            {"total_length": 68, "start": 68, "end": 68},
            {"total_length": 68, "start": 0, "end": 69},
            {"total_length": 68, "start": -1, "end": 1},
        ):
            with self.assertRaises(ValueError):
                plan_prefill(kind="linear_attention", **kwargs)

    def test_no_device_imports(self):
        # Repository pytest fixtures may already import device packages. Isolate
        # the production module import, including its transitive dependencies.
        # -I/-S exclude inherited Python paths and site initialization; only the
        # explicit checkout root is added, so this check also works outside cwd.
        probe = (
            "import sys; sys.path.insert(0, sys.argv[1]); "
            "__import__(sys.argv[2]); "
            "found = sorted(name for name in sys.modules "
            "if name.split('.')[0] in ('ttnn', 'torch', 'vllm')); "
            "assert not found, 'Host-only module imported device packages: ' + repr(found)"
        )
        result = subprocess.run(
            [
                sys.executable,
                "-I",
                "-S",
                "-c",
                probe,
                str(Path(__file__).resolve().parents[4]),
                "models.autoports.ornith_ai_ornith_1_5_9b.tt.prefill_plan",
            ],
            capture_output=True,
            check=False,
            text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        self.assertEqual(max(LINEAR_BUCKETS), CHUNK_CAPS["linear_attention"])
        self.assertEqual(max(FULL_PREFILL_BUCKETS), CHUNK_CAPS["full_attention"])


if __name__ == "__main__":
    unittest.main()
