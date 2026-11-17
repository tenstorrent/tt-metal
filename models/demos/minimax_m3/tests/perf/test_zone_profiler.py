# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device-free tests for the MiniMax-M3 side of the zone profiler.

The mechanism (attribution, truncation detection, leaf detection, per-layer aggregation) is tested
once in models/demos/common/prefill/tests/test_zone_profiling.py. What is pinned here is the M3
contract with it: the layer tag tt/layer.py emits, which zones count as communication / memory, and
the env-var names the wrapper script exports. Building a real Model needs 32 devices, so the
layer-selection test covers the config contract (the part that can corrupt KV-cache addressing)
rather than the build.

    pytest models/demos/minimax_m3/tests/perf/test_zone_profiler.py
"""

from pathlib import Path

import pytest

from models.demos.common.prefill.profiling import parse_zone_perf as P
from models.demos.minimax_m3.utils.profiler_utils import SPEC

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "run_prefill_profile.sh"


class TestSpec:
    def test_layer_classes_match_the_layer_tags(self):
        # tt/layer.py: f"layer{layer_idx:02d}_{'sparse' if is_sparse else 'dense'}"
        assert SPEC.class_keys == ("dense", "sparse")
        assert P.layer_class(f"{SPEC.root_zone}/layer00_dense/attn/qkv_proj") == ("dense", "attn/qkv_proj", 0)
        assert P.layer_class(f"{SPEC.root_zone}/layer07_sparse/mlp/combine") == ("sparse", "mlp/combine", 7)
        assert P.relative_path(f"{SPEC.root_zone}/layer05_sparse") == "sparse:(layer total)"

    def test_full_model_is_60_layers(self):
        assert SPEC.full_model_layers == 60

    def test_wrapper_script_exports_the_spec_env_vars(self):
        text = SCRIPT.read_text()
        assert f"export {SPEC.zones_env}=1" in text
        assert SPEC.level_env in text


class TestCategorization:
    """SPEC.cat() drives the compute/comm/memory split — the headline number of the report."""

    @pytest.mark.parametrize(
        "rel",
        [
            "attn/ccl_out_allreduce",
            "attn/ag_kv",
            "attn/ag_index_k",
            "mlp/tp_allreduce",
            "mlp/tp_allgather",
            "mlp/dispatch",
            "mlp/combine",
            "mlp/moe_reduce",
            "mlp/pre_dispatch_allgather",
            "mlp/shared_expert/tp_allreduce",
        ],
    )
    def test_collectives_are_comm(self, rel):
        assert SPEC.cat(rel) == "comm"

    @pytest.mark.parametrize("rel", ["attn/kv_write", "attn/kv_write/k", "attn/index_k_write"])
    def test_cache_writes_are_memory(self, rel):
        assert SPEC.cat(rel) == "memory"

    @pytest.mark.parametrize(
        "rel", ["attn/ring_joint_sdpa", "attn/sparse_sdpa", "attn/indexer", "mlp/experts_mm", "mlp/shared_expert"]
    )
    def test_compute_zones(self, rel):
        # ring_joint_sdpa fuses its ring CCL with the attention compute in one op: reported as compute.
        assert SPEC.cat(rel) == "compute"


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
