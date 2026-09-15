# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device-free tests for the zone profiler's contracts.

Everything here runs on host: the pieces that can silently produce a *wrong report* rather than an
error are the ones worth pinning. Building a real Model needs 32 devices, so the layer-selection test
covers the config contract (the part that can corrupt KV-cache addressing) rather than the build.

    pytest models/demos/minimax_m3/tests/perf/test_zone_profiler.py
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import parse_zone_perf as P  # noqa: E402
import visualize_zones as V  # noqa: E402


class TestLeafDetection:
    """A zone is a parent only when a descendant is present in THIS capture.

    A static parent list silently dropped `attn/cache_read` from the totals at LEVEL=2, where its
    deshard/slice children are suppressed and it is therefore a leaf.
    """

    LEVEL2 = {
        "(layer total)",
        "attn",
        "mlp",
        "attn/cache_read",
        "attn/sparse_sdpa",
        "mlp/experts_mm",
        "mlp/shared_expert",
        "mlp/shared_expert/tp_allreduce",
    }
    LEVEL3 = LEVEL2 | {"attn/cache_read/deshard", "attn/cache_read/slice"}

    def test_cache_read_is_a_leaf_when_its_children_are_suppressed(self):
        assert "attn/cache_read" not in V.parent_rels(self.LEVEL2)

    def test_cache_read_is_a_parent_when_its_children_are_captured(self):
        assert "attn/cache_read" in V.parent_rels(self.LEVEL3)

    def test_real_parents_are_always_excluded(self):
        for level in (self.LEVEL2, self.LEVEL3):
            parents = V.parent_rels(level)
            assert {"attn", "mlp", "mlp/shared_expert", "(layer total)"} <= parents

    def test_leaves_and_parents_partition_the_zones(self):
        # Nothing may be both, and nothing may be neither — otherwise time is double-counted or lost.
        for level in (self.LEVEL2, self.LEVEL3):
            parents = V.parent_rels(level)
            leaves = level - parents
            assert parents | leaves == level
            assert not (parents & leaves)


class TestZoneAccumulator:
    """Attribution: ops belong to the innermost open zone and every enclosing one, and only the
    profiled chunk is reported."""

    @staticmethod
    def _sig(name):
        return {"OP CODE": name, "OP TYPE": "signpost"}

    @staticmethod
    def _op(ns, dev=0):
        return {
            "OP CODE": "Matmul",
            "OP TYPE": P.DEVICE_OP_TYPE,
            "DEVICE ID": dev,
            P.DURATION_COL: ns,
        }

    def _feed(self, rows):
        acc = P.ZoneAccumulator()
        for r in rows:
            acc.feed(r, {})
        return acc

    def test_ops_outside_the_root_zone_are_ignored(self):
        # Warmup and cache-prefix ops share the CSV with the profiled chunk; they must not be counted.
        acc = self._feed([self._op(1_000_000)])
        assert acc.rows_in_root == 0

    def test_op_is_charged_to_every_enclosing_zone(self):
        acc = self._feed(
            [
                self._sig(f"{P.ZONE_START} {P.ROOT_ZONE}"),
                self._sig(f"{P.ZONE_START} layer03_sparse"),
                self._sig(f"{P.ZONE_START} attn"),
                self._op(2_000_000),
                self._sig(f"{P.ZONE_END} attn"),
                self._sig(f"{P.ZONE_END} layer03_sparse"),
                self._sig(f"{P.ZONE_END} {P.ROOT_ZONE}"),
            ]
        )
        for path in (P.ROOT_ZONE, f"{P.ROOT_ZONE}/layer03_sparse", f"{P.ROOT_ZONE}/layer03_sparse/attn"):
            assert acc.stats[(path, 0)]["ns"] == 2_000_000, path

    def test_unmatched_end_marker_is_counted_not_fatal(self):
        # A truncated capture must degrade, not crash: the report says so instead of dying.
        acc = self._feed([self._sig(f"{P.ZONE_END} never_opened")])
        assert acc.unmatched_ends == 1

    def test_non_device_ops_are_flagged_as_host_work(self):
        acc = self._feed(
            [
                self._sig(f"{P.ZONE_START} {P.ROOT_ZONE}"),
                {"OP CODE": "Fallback", "OP TYPE": "python_fallback", "DEVICE ID": 0, P.DURATION_COL: None},
                self._sig(f"{P.ZONE_END} {P.ROOT_ZONE}"),
            ]
        )
        assert acc.host_ops, "a CPU fallback inside the forward must be reported"


class TestLayerIndicesContract:
    """`layer_indices` sizes the model; `num_layers` sizes the KV cache. If they disagree, a layer
    addresses past the per-user cache stride."""

    def _config(self, **kw):
        from models.demos.minimax_m3.tt.tt_prefill_runtime import TtPrefillRuntimeConfig

        base = dict(num_layers=2, max_seq_len=10240, chunk_size=5120)
        base.update(kw)
        return TtPrefillRuntimeConfig(**base)

    def test_matching_lengths_are_accepted(self):
        cfg = self._config(layer_indices=[0, 3])
        assert len(cfg.layer_indices) == cfg.num_layers

    def test_default_is_contiguous(self):
        assert self._config().layer_indices is None

    @pytest.mark.parametrize("indices", [[0, 3, 4], [0]])
    def test_mismatched_length_is_rejected(self, indices, expect_error):
        from models.demos.minimax_m3.tt.tt_prefill_runtime import TtPrefillRuntime

        cfg = self._config(layer_indices=indices)
        with expect_error(AssertionError, "layer_indices"):
            TtPrefillRuntime.__init__(object.__new__(TtPrefillRuntime), None, None, {}, cfg)
