# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device-free tests for the GPT-OSS side of the zone profiler.

The mechanism (zone gating, signpost wire format, attribution, truncation detection, leaf detection,
per-layer aggregation) is tested once in models/demos/common/prefill/tests/test_zone_profiling.py. What
is pinned here is the GPT-OSS contract with it: the layer tag tt/layer.py emits, which zones count as
communication / memory, the env-var names the wrapper script exports, and the harness's chunk plan.

    pytest models/demos/gpt_oss_d_p/tests/perf/test_zone_profiler.py
"""

import importlib
import os
from pathlib import Path

import pytest

from models.demos.common.prefill.profiling import parse_zone_perf as P
from models.demos.gpt_oss_d_p.utils.profiler_utils import SPEC

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "run_prefill_profile.sh"


class TestSpec:
    def test_layer_classes_match_the_layer_tags(self):
        # tt/layer.py: f"layer{layer_idx:02d}_{'sliding' if self.self_attn.is_sliding else 'full'}"
        assert SPEC.class_keys == ("sliding", "full")
        assert P.layer_class(f"{SPEC.root_zone}/layer00_sliding/attn/qkv_proj") == ("sliding", "attn/qkv_proj", 0)
        assert P.layer_class(f"{SPEC.root_zone}/layer07_full/mlp/combine") == ("full", "mlp/combine", 7)
        assert P.layer_class(f"{SPEC.root_zone}/layer11_full") == ("full", "", 11)
        assert P.relative_path(f"{SPEC.root_zone}/layer04_sliding/attn/sdpa") == "sliding:attn/sdpa"

    def test_full_model_is_36_layers_half_each(self):
        assert SPEC.full_model_layers == 36
        assert [c.full_model_count for c in SPEC.layer_classes] == [18, 18]

    def test_wrapper_script_exports_the_spec_env_vars(self):
        text = SCRIPT.read_text()
        assert f"export {SPEC.zones_env}=1" in text
        assert SPEC.level_env in text


class TestCategorization:
    """SPEC.cat() drives the compute/comm/memory split — the headline number of the report."""

    @pytest.mark.parametrize(
        "rel",
        [
            "attn/ag_qkv",
            "attn/sdpa_reduce_scatter",
            "attn/ccl_out_allreduce",
            "attn/ccl_out_allgather",
            "mlp/tp_allgather",
            "mlp/dispatch",
            "mlp/combine",
            "mlp/moe_reduce",
            "mlp/pre_dispatch_allgather",
        ],
    )
    def test_collectives_are_comm(self, rel):
        assert SPEC.cat(rel) == "comm"

    @pytest.mark.parametrize("rel", ["attn/kv_write", "defrag_move"])
    def test_cache_traffic_is_memory(self, rel):
        assert SPEC.cat(rel) == "memory"

    def test_ring_sdpa_is_compute(self):
        # The cache-backed ring SDPA fuses its SP ring CCL with the attention compute in one device
        # op, so it is reported as compute (its comm share is not separable) — see profiler_utils.py.
        assert SPEC.cat("attn/ring_joint_sdpa") == "compute"

    @pytest.mark.parametrize(
        "rel", ["attn/qkv_proj", "attn/sdpa", "attn/o_proj", "mlp/experts_mm", "mlp/router_topk", f"mlp/{P.SELF}"]
    )
    def test_matmuls_and_glue_are_compute(self, rel):
        assert SPEC.cat(rel) == "compute"


class TestHarness:
    """profile_prefill.py: importing it must have no side effects, and its chunk plan sizes the KV
    cache the profiled chunk attends — a wrong plan profiles the wrong case."""

    @staticmethod
    def _harness():
        return importlib.import_module("models.demos.gpt_oss_d_p.tests.perf.profile_prefill")

    def test_import_does_not_arm_the_profiler(self, monkeypatch):
        # The env flags belong to main(): a test that imports the harness must not turn on zones and
        # the device profiler for every later test in the same pytest session.
        monkeypatch.delenv("GPTOSS_PROFILE_ZONES", raising=False)
        monkeypatch.delenv("TT_METAL_DEVICE_PROFILER", raising=False)
        importlib.reload(self._harness())
        assert "GPTOSS_PROFILE_ZONES" not in os.environ
        assert "TT_METAL_DEVICE_PROFILER" not in os.environ

    def test_cache_rounds_down_to_whole_chunks(self):
        assert self._harness().plan(8192, 25000) == (4, 24576, 32768)

    def test_zero_cache_is_one_shot(self):
        assert self._harness().plan(8192, 0) == (1, 0, 8192)

    def test_misaligned_chunk_is_rejected(self, expect_error):
        # chunk/sp must split across the 64 MoE routing cores (galaxy_prefill_kv_pcc.chunk_alignment).
        with expect_error(AssertionError, "must be a multiple"):
            self._harness().plan(8000, 0)
