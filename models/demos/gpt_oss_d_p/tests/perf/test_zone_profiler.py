# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device-free tests for the GPT-OSS zone profiler's contracts.

Everything here runs on host: the pieces that can silently produce a *wrong report* rather than an
error are the ones worth pinning. Mirrors ``minimax_m3/tests/perf/test_zone_profiler.py``.

    pytest models/demos/gpt_oss_d_p/tests/perf/test_zone_profiler.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import parse_zone_perf as P  # noqa: E402
import visualize_zones as V  # noqa: E402


class TestLeafDetection:
    """A zone is a parent only when a descendant is present in THIS capture.

    Whether a zone is a parent depends on the capture's LEVEL (suppressed children make it a leaf),
    so the exclusion set must be computed from the capture, never from a static list — a static list
    silently drops leaf time from the totals.
    """

    LEVEL2 = {
        "(layer total)",
        "attn",
        "mlp",
        "attn/ring_joint_sdpa",
        "attn/kv_write",
        "mlp/experts_mm",
        "mlp/dispatch",
    }
    LEVEL3 = LEVEL2 | {"attn/qkv_proj", "attn/rope", "mlp/routing_setup"}

    def test_zone_without_captured_children_is_a_leaf(self):
        assert "attn/ring_joint_sdpa" not in V.parent_rels(self.LEVEL2)
        assert "mlp/dispatch" not in V.parent_rels(self.LEVEL3)

    def test_real_parents_are_always_excluded(self):
        for level in (self.LEVEL2, self.LEVEL3):
            parents = V.parent_rels(level)
            assert {"attn", "mlp", "(layer total)"} <= parents

    def test_leaves_and_parents_partition_the_zones(self):
        # Nothing may be both, and nothing may be neither — otherwise time is double-counted or lost.
        for level in (self.LEVEL2, self.LEVEL3):
            parents = V.parent_rels(level)
            leaves = level - parents
            assert parents | leaves == level
            assert not (parents & leaves)


class TestCategorization:
    """cat() drives the compute/comm/memory split — the headline number of the report."""

    def test_collectives_are_comm(self):
        for rel in (
            "attn/ag_qkv",
            "attn/sdpa_reduce_scatter",
            "attn/ccl_out_allreduce",
            "attn/ccl_out_allgather",
            "mlp/tp_allgather",
            "mlp/dispatch",
            "mlp/combine",
            "mlp/moe_reduce",
            "mlp/pre_dispatch_allgather",
        ):
            assert V.cat(rel) == "comm", rel

    def test_kv_cache_traffic_is_memory(self):
        assert V.cat("attn/kv_write") == "memory"

    def test_ring_sdpa_is_compute(self):
        # The cache-backed ring SDPA fuses its SP ring CCL with the attention compute in one device
        # op, so it is reported as compute (its comm share is not separable) — see visualize_zones.py.
        assert V.cat("attn/ring_joint_sdpa") == "compute"

    def test_matmuls_are_compute(self):
        for rel in ("attn/qkv_proj", "attn/sdpa", "attn/o_proj", "mlp/experts_mm", "mlp/router_topk"):
            assert V.cat(rel) == "compute", rel


class TestLayerClassParsing:
    """The layerNN_{sliding|full} tag is the contract between layer.py and the parser's aggregation."""

    def test_sliding_layer(self):
        assert P.layer_class(f"{P.ROOT_ZONE}/layer00_sliding/attn/qkv_proj") == ("sliding", "attn/qkv_proj", 0)

    def test_full_layer(self):
        assert P.layer_class(f"{P.ROOT_ZONE}/layer07_full/mlp/combine") == ("full", "mlp/combine", 7)

    def test_layer_total(self):
        assert P.layer_class(f"{P.ROOT_ZONE}/layer11_full") == ("full", "", 11)

    def test_non_layer_zone(self):
        assert P.layer_class("profiled_chunk") is None

    def test_relative_path_collapses_layer_index(self):
        assert P.relative_path("profiled_chunk/layer04_sliding/attn/sdpa") == "sliding:attn/sdpa"
        assert P.relative_path("profiled_chunk/layer05_full") == "full:(layer total)"


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
                self._sig(f"{P.ZONE_START} layer03_full"),
                self._sig(f"{P.ZONE_START} attn"),
                self._op(2_000_000),
                self._sig(f"{P.ZONE_END} attn"),
                self._sig(f"{P.ZONE_END} layer03_full"),
                self._sig(f"{P.ZONE_END} {P.ROOT_ZONE}"),
            ]
        )
        for path in (P.ROOT_ZONE, f"{P.ROOT_ZONE}/layer03_full", f"{P.ROOT_ZONE}/layer03_full/attn"):
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


class TestChunkPlan:
    """plan() sizes the KV cache the profiled chunk attends; a wrong plan profiles the wrong case."""

    def _plan(self, chunk, cache):
        import profile_prefill as H

        return H.plan(chunk, cache)

    def test_cache_rounds_down_to_whole_chunks(self):
        n_chunks, cache, total = self._plan(8192, 25000)
        assert (n_chunks, cache, total) == (4, 24576, 32768)

    def test_zero_cache_is_one_shot(self):
        n_chunks, cache, total = self._plan(8192, 0)
        assert (n_chunks, cache, total) == (1, 0, 8192)

    def test_misaligned_chunk_is_rejected(self, expect_error):
        # chunk/sp must split across the 64 MoE routing cores (see galaxy_prefill_kv_pcc.plan).
        with expect_error(AssertionError, "must be a multiple"):
            self._plan(8000, 0)
