# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import dataclasses
from types import SimpleNamespace

import pytest
import torch

from ttnn._experimental.auto_config import matmul as auto_matmul


@pytest.fixture(autouse=True)
def clear_runtime_records():
    auto_matmul.AutoMatmulCache.clear_runtime()
    yield
    auto_matmul.AutoMatmulCache.clear_runtime()


def _make_signature():
    return auto_matmul.AutoMatmulSignature(
        arch="wormhole_b0",
        device_count=1,
        mesh_shape=(1,),
        is_linear=False,
        transpose_a=False,
        transpose_b=False,
        activation=None,
        output_memory_config="DRAM_MEMORY_CONFIG",
        output_dtype="bfloat16",
        input_tensor_a={
            "shape": [1, 64, 96],
            "dtype": "bfloat16",
            "layout": "Layout.TILE",
            "tile": {"tile_shape": [32, 32], "transpose_of_faces": "False"},
            "memory_config": "DRAM_MEMORY_CONFIG",
            "topology": None,
        },
        input_tensor_b={
            "shape": [96, 128],
            "dtype": "bfloat16",
            "layout": "Layout.TILE",
            "tile": {"tile_shape": [32, 32], "transpose_of_faces": "False"},
            "memory_config": "DRAM_MEMORY_CONFIG",
            "topology": None,
        },
        bias=None,
        m=64,
        k=96,
        n=128,
    )


def _make_linear_signature(*, a_shape=(256, 1024), b_shape=(1024, 512), bias_shape=(512,)):
    signature = _make_signature()
    m, k, n = auto_matmul._extract_mkn(a_shape, b_shape, False, False)
    return dataclasses.replace(
        signature,
        is_linear=True,
        input_tensor_a={**signature.input_tensor_a, "shape": list(a_shape)},
        input_tensor_b={**signature.input_tensor_b, "shape": list(b_shape)},
        bias={
            "shape": list(bias_shape),
            "dtype": "bfloat16",
            "layout": "Layout.TILE",
            "memory_config": "DRAM_MEMORY_CONFIG",
            "topology": None,
        },
        m=m,
        k=k,
        n=n,
    )


def _make_distributed_signature(*, lhs_shard_dim=None, rhs_shard_dim=None):
    lhs_topology = None
    rhs_topology = None
    if lhs_shard_dim is not None:
        lhs_topology = {
            "placements": [f"PlacementShard({lhs_shard_dim})"],
            "placement_kinds": ["shard"],
            "shard_dims": [lhs_shard_dim],
            "normalized_shard_dims": [lhs_shard_dim],
            "distribution_shape": [8],
        }
    if rhs_shard_dim is not None:
        rhs_topology = {
            "placements": [f"PlacementShard({rhs_shard_dim})"],
            "placement_kinds": ["shard"],
            "shard_dims": [rhs_shard_dim],
            "normalized_shard_dims": [rhs_shard_dim],
            "distribution_shape": [8],
        }
    return auto_matmul.AutoMatmulSignature(
        arch="wormhole_b0",
        device_count=8,
        mesh_shape=(8,),
        is_linear=False,
        transpose_a=False,
        transpose_b=False,
        activation=None,
        output_memory_config="DRAM_MEMORY_CONFIG",
        output_dtype="bfloat16",
        input_tensor_a={
            "shape": [1, 1, 32, 64],
            "dtype": "bfloat16",
            "layout": "Layout.TILE",
            "tile": {"tile_shape": [32, 32], "transpose_of_faces": "False"},
            "memory_config": "DRAM_MEMORY_CONFIG",
            "topology": lhs_topology,
        },
        input_tensor_b={
            "shape": [1, 1, 64, 128],
            "dtype": "bfloat16",
            "layout": "Layout.TILE",
            "tile": {"tile_shape": [32, 32], "transpose_of_faces": "False"},
            "memory_config": "DRAM_MEMORY_CONFIG",
            "topology": rhs_topology,
        },
        bias=None,
        m=32,
        k=64,
        n=128,
    )


def test_extract_mkn_respects_transposes():
    assert auto_matmul._extract_mkn((2, 64, 96), (96, 128), False, False) == (128, 96, 128)
    assert auto_matmul._extract_mkn((4, 16), (32, 16), False, True) == (4, 16, 32)
    assert auto_matmul._extract_mkn((4, 16), (4, 32), True, False) == (16, 4, 32)


def test_extract_mkn_uses_broadcasted_batch_shape():
    assert auto_matmul._extract_mkn((1, 32, 64), (8, 64, 128), False, False) == (256, 64, 128)


def test_linear_bias_broadcast_that_changes_output_shape_disables_minimal_candidates():
    signature = _make_linear_signature(bias_shape=(1, 1, 512))
    bias = SimpleNamespace(shape=(1, 1, 512), dtype="bfloat16", layout="Layout.TILE")

    assert auto_matmul._broadcast_shape((256, 512), (1, 1, 512)) == (1, 256, 512)
    assert not auto_matmul._minimal_matmul_preserves_linear_bias_shape(signature)
    assert not auto_matmul._can_use_minimal_matmul_common(signature, bias)


@pytest.mark.parametrize("bias_shape", [(512,), (1, 512)])
def test_linear_bias_broadcast_that_preserves_output_shape_allows_minimal_candidates(bias_shape):
    signature = _make_linear_signature(bias_shape=bias_shape)
    bias = SimpleNamespace(shape=bias_shape, dtype="bfloat16", layout="Layout.TILE")

    assert auto_matmul._minimal_matmul_preserves_linear_bias_shape(signature)
    assert auto_matmul._can_use_minimal_matmul_common(signature, bias)


def test_tiny_tile_shapes_disable_minimal_candidates():
    signature = _make_signature()
    signature = dataclasses.replace(
        signature,
        input_tensor_a={**signature.input_tensor_a, "tile": {"tile_shape": [8, 32], "transpose_of_faces": "False"}},
        input_tensor_b={**signature.input_tensor_b, "tile": {"tile_shape": [32, 16], "transpose_of_faces": "False"}},
    )

    assert not auto_matmul._uses_standard_tile_shape(signature.input_tensor_a)
    assert not auto_matmul._uses_standard_tile_shape(signature.input_tensor_b)
    assert not auto_matmul._can_use_minimal_matmul_common(signature, bias=None)


def test_standard_tile_shapes_allow_minimal_candidates():
    signature = _make_signature()

    assert auto_matmul._uses_standard_tile_shape(signature.input_tensor_a)
    assert auto_matmul._uses_standard_tile_shape(signature.input_tensor_b)
    assert auto_matmul._can_use_minimal_matmul_common(signature, bias=None)


def test_cache_round_trip_and_force_retune(monkeypatch, tmp_path):
    monkeypatch.setenv("TTNN_AUTO_MATMUL_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("TTNN_AUTO_MATMUL_VERSION", "unit-test-version")
    signature = _make_signature()
    cache = auto_matmul.AutoMatmulCache()
    payload = {
        "version": cache.version,
        "winner": {"kind": "default_matmul"},
        "candidate_timings_us": [],
        "recommendations": [],
    }

    saved_path = cache.save(signature, payload)
    assert saved_path == cache.path_for(signature)
    assert cache.load(signature) == payload

    monkeypatch.setenv("TTNN_AUTO_MATMUL_FORCE_RETUNE", "1")
    assert cache.load(signature) is None


def test_cache_key_stable():
    lhs = _make_signature()
    rhs = _make_signature()
    assert lhs.cache_key == rhs.cache_key
    assert len(lhs.cache_key) == 64


def test_candidate_helpers_cover_small_and_large_tiles():
    assert auto_matmul._get_mn_block_candidates(1) == [1]
    assert auto_matmul._get_k_block_candidates(1) == [1]
    assert 8 in auto_matmul._get_mn_block_candidates(16)
    assert 4 in auto_matmul._get_k_block_candidates(16)
    assert auto_matmul._pick_subblock(8, 8) == (2, 2)
    assert auto_matmul._pick_subblock(3, 5) == (3, 1)


def test_host_tensor_cache_name_tracks_tensor_contents(monkeypatch, tmp_path):
    monkeypatch.setenv("TTNN_AUTO_MATMUL_CACHE_DIR", str(tmp_path))
    first = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    second = first.clone()
    second[0, 0] = -1.0

    first_path = auto_matmul._host_tensor_cache_name("rhs", first, "bfloat16")
    second_path = auto_matmul._host_tensor_cache_name("rhs", second, "bfloat16")

    assert first_path != second_path
    assert first_path.parent == second_path.parent == tmp_path / "host_tensors"


def test_explicit_override_detects_compute_kernel_and_output_tile():
    assert auto_matmul._has_explicit_override({"compute_kernel_config": object()})
    assert auto_matmul._has_explicit_override({"output_tile": object()})
    assert not auto_matmul._has_explicit_override({"dtype": "bfloat16"})


def test_env_override_skips_git_hash(monkeypatch):
    monkeypatch.setenv("TTNN_AUTO_MATMUL_VERSION", "override-version")
    assert auto_matmul._get_default_version() == "override-version"


def test_default_version_falls_back_to_ci_sha_when_git_hash_unavailable(monkeypatch):
    monkeypatch.delenv("TTNN_AUTO_MATMUL_VERSION", raising=False)
    monkeypatch.setenv("GITHUB_SHA", "0123456789abcdef")

    def fake_import_module(name):
        assert name == "ttnn.model_preprocessing"

        def raise_git_hash():
            raise RuntimeError("Couldn't get git hash!")

        return SimpleNamespace(git_hash=raise_git_hash)

    monkeypatch.setattr(auto_matmul.importlib, "import_module", fake_import_module)

    assert auto_matmul._get_default_version() == "0123456789abcdef"


def test_default_version_falls_back_to_unknown_without_git_or_package_metadata(monkeypatch):
    monkeypatch.delenv("TTNN_AUTO_MATMUL_VERSION", raising=False)
    monkeypatch.delenv("GITHUB_SHA", raising=False)
    monkeypatch.delenv("CI_COMMIT_SHA", raising=False)
    monkeypatch.delenv("BUILD_SOURCEVERSION", raising=False)
    monkeypatch.delenv("GIT_REF", raising=False)

    def fake_import_module(name):
        assert name == "ttnn.model_preprocessing"

        def raise_git_hash():
            raise RuntimeError("Couldn't get git hash!")

        return SimpleNamespace(git_hash=raise_git_hash)

    monkeypatch.setattr(auto_matmul.importlib, "import_module", fake_import_module)
    monkeypatch.setattr(
        auto_matmul.importlib_metadata,
        "version",
        lambda package_name: (_ for _ in ()).throw(RuntimeError(f"Missing package metadata for {package_name}")),
    )

    assert auto_matmul._get_default_version() == "unknown"


def test_infer_distributed_plan_for_all_gather():
    signature = _make_distributed_signature(lhs_shard_dim=3)
    plan = auto_matmul._infer_distributed_plan(signature)
    assert plan.kind == "gather_before_matmul"
    assert plan.collective_dim == 3
    assert plan.cluster_axis == 0
    assert plan.distribution_factor == 8


def test_infer_distributed_plan_for_reduce_scatter():
    signature = _make_distributed_signature(rhs_shard_dim=2)
    plan = auto_matmul._infer_distributed_plan(signature)
    assert plan.kind == "matmul_before_reduce_scatter"
    assert plan.collective_dim == 3
    assert plan.cluster_axis == 0
    assert plan.distribution_factor == 8


def test_infer_distributed_plan_marks_unsupported():
    signature = _make_distributed_signature(lhs_shard_dim=2)
    plan = auto_matmul._infer_distributed_plan(signature)
    assert plan.kind == "unsupported"


def test_supported_distributed_candidates_skip_local_default(monkeypatch):
    signature = _make_distributed_signature(rhs_shard_dim=2)
    prepared = SimpleNamespace()
    reduce_scatter_candidate = auto_matmul.Candidate(
        descriptor={"kind": "matmul_then_reduce_scatter"}, run=lambda: None
    )

    monkeypatch.setattr(
        auto_matmul,
        "_build_default_candidate",
        lambda **kwargs: auto_matmul.Candidate(descriptor={"kind": "default_matmul"}, run=lambda: None),
    )
    monkeypatch.setattr(
        auto_matmul,
        "_build_matmul_then_reduce_scatter_candidate",
        lambda *args, **kwargs: reduce_scatter_candidate,
    )
    monkeypatch.setattr(auto_matmul, "_build_minimal_then_reduce_scatter_candidates", lambda *args, **kwargs: [])
    monkeypatch.setattr(auto_matmul, "_build_minimal_matmul_reduce_scatter_candidates", lambda *args, **kwargs: [])

    candidates = auto_matmul._build_candidates(signature, prepared, {}, base_operation=None)

    assert [candidate.descriptor["kind"] for candidate in candidates] == ["matmul_then_reduce_scatter"]


def test_unsupported_distributed_candidates_do_not_fall_back_to_local_default(monkeypatch):
    signature = _make_distributed_signature(lhs_shard_dim=2)
    prepared = SimpleNamespace()

    monkeypatch.setattr(
        auto_matmul,
        "_build_default_candidate",
        lambda **kwargs: auto_matmul.Candidate(descriptor={"kind": "default_matmul"}, run=lambda: None),
    )
    monkeypatch.setattr(auto_matmul, "_build_local_minimal_candidates", lambda *args, **kwargs: [])

    candidates = auto_matmul._build_candidates(signature, prepared, {}, base_operation=None)

    assert candidates == []


def test_blackhole_reduce_scatter_candidates_skip_unstable_fused_path():
    signature = dataclasses.replace(_make_distributed_signature(rhs_shard_dim=2), arch="blackhole")
    prepared = SimpleNamespace(bias=None)

    candidates = auto_matmul._build_minimal_matmul_reduce_scatter_candidates(
        signature,
        prepared,
        {},
        auto_matmul._infer_distributed_plan(signature),
    )

    assert candidates == []


def test_select_candidate_chooses_fastest_and_persists_record(monkeypatch, tmp_path):
    monkeypatch.setenv("TTNN_AUTO_MATMUL_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("TTNN_AUTO_MATMUL_VERSION", "unit-test-version")
    auto_matmul.AutoMatmulCache.clear_runtime()

    signature = _make_signature()
    prepared = SimpleNamespace(
        input_tensor_a=SimpleNamespace(device=lambda: "device"),
        staged_rhs_from_host=False,
        staged_bias_from_host=False,
    )
    candidates = [
        auto_matmul.Candidate(descriptor={"kind": "default_matmul"}, run=lambda: None),
        auto_matmul.Candidate(descriptor={"kind": "minimal_matmul"}, run=lambda: None),
    ]

    monkeypatch.setattr(auto_matmul, "_build_candidates", lambda *args, **kwargs: candidates)
    monkeypatch.setattr(
        auto_matmul,
        "_benchmark_candidate",
        lambda candidate, device: (
            (20.0, [20.0, 20.0, 20.0], "trace")
            if candidate.descriptor["kind"] == "default_matmul"
            else (10.0, [10.0, 10.0, 10.0], "trace")
        ),
    )
    monkeypatch.setattr(auto_matmul, "_make_recommendations", lambda *args, **kwargs: ["unit-test"])

    selection = auto_matmul._select_candidate(
        signature,
        prepared,
        {},
        base_operation=None,
        allow_tuning=True,
    )

    assert selection["winner"] == {"kind": "minimal_matmul"}
    assert selection["candidate"].descriptor == {"kind": "minimal_matmul"}
    assert [entry["descriptor"]["kind"] for entry in selection["candidate_timings_us"]] == [
        "default_matmul",
        "minimal_matmul",
    ]

    cached_record = auto_matmul.AutoMatmulCache().load(signature)
    assert cached_record is not None
    assert cached_record["winner"] == {"kind": "minimal_matmul"}


def test_select_candidate_uses_runtime_winner_when_tuning_is_disabled(monkeypatch, tmp_path):
    monkeypatch.setenv("TTNN_AUTO_MATMUL_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("TTNN_AUTO_MATMUL_VERSION", "unit-test-version")
    monkeypatch.setenv("TTNN_AUTO_MATMUL_FORCE_RETUNE", "1")
    auto_matmul.AutoMatmulCache.clear_runtime()

    signature = _make_signature()
    prepared = SimpleNamespace()
    cache = auto_matmul.AutoMatmulCache()
    runtime_record = {
        "winner": {"kind": "matmul_then_reduce_scatter"},
        "candidate_timings_us": [{"descriptor": {"kind": "matmul_then_reduce_scatter"}, "status": "ok"}],
        "recommendations": ["runtime winner"],
    }
    cache.save_runtime(signature, runtime_record)

    selection = auto_matmul._select_candidate(
        signature,
        prepared,
        {},
        base_operation=None,
        allow_tuning=False,
    )

    assert selection["winner"] == runtime_record["winner"]
    assert selection["candidate_timings_us"] == runtime_record["candidate_timings_us"]
    assert selection["recommendations"] == runtime_record["recommendations"]
    assert selection["candidate"] is None


def test_dispatch_matmul_stages_host_rhs_even_when_auto_config_disabled(monkeypatch):
    class FakeTensor:
        shape = (4, 4)

        def device(self):
            return "device"

    class FailingCache:
        def __init__(self):
            raise AssertionError("cache should not be constructed when auto_config is disabled")

    prepared = auto_matmul.PreparedMatmulInputs(
        input_tensor_a=FakeTensor(),
        input_tensor_b="staged-rhs",
        bias="staged-bias",
        staged_rhs_from_host=True,
        staged_bias_from_host=True,
    )
    calls = []

    monkeypatch.setattr(auto_matmul, "_ttnn", lambda: SimpleNamespace(Tensor=FakeTensor))
    monkeypatch.setattr(auto_matmul, "_prepare_inputs", lambda *args, **kwargs: prepared)
    monkeypatch.setattr(auto_matmul, "AutoMatmulCache", FailingCache)

    def base_operation(lhs, rhs, **kwargs):
        calls.append((lhs, rhs, kwargs))
        return "base-result"

    result = auto_matmul.dispatch_matmul(
        base_operation=base_operation,
        input_tensor_a=FakeTensor(),
        input_tensor_b=torch.ones((4, 4), dtype=torch.float32),
        bias=torch.ones((4,), dtype=torch.float32),
        is_linear=True,
        auto_config=False,
    )

    assert result == "base-result"
    assert calls == [(prepared.input_tensor_a, prepared.input_tensor_b, {"bias": prepared.bias})]


def test_dispatch_matmul_bypasses_selector_for_unsupported_distributed_plan(monkeypatch):
    class FakeTensor:
        shape = (4, 4)

        def device(self):
            return "device"

    prepared = auto_matmul.PreparedMatmulInputs(
        input_tensor_a=FakeTensor(),
        input_tensor_b=FakeTensor(),
        bias=None,
    )
    signature = _make_distributed_signature(lhs_shard_dim=2)
    calls = []

    monkeypatch.setattr(auto_matmul, "_ttnn", lambda: SimpleNamespace(Tensor=FakeTensor))
    monkeypatch.setattr(auto_matmul, "_prepare_inputs", lambda *args, **kwargs: prepared)
    monkeypatch.setattr(auto_matmul, "_build_signature", lambda *args, **kwargs: signature)
    monkeypatch.setattr(
        auto_matmul,
        "_select_candidate",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("selector should be bypassed")),
    )

    def base_operation(lhs, rhs, **kwargs):
        calls.append((lhs, rhs, kwargs))
        return "base-result"

    result = auto_matmul.dispatch_matmul(
        base_operation=base_operation,
        input_tensor_a=FakeTensor(),
        input_tensor_b=FakeTensor(),
        bias=None,
        is_linear=False,
        auto_config=True,
    )

    assert result == "base-result"
    assert calls == [(prepared.input_tensor_a, prepared.input_tensor_b, {})]


def test_dispatch_matmul_bypasses_selector_when_graph_report_is_enabled(monkeypatch):
    class FakeTensor:
        shape = (4, 4)

        def device(self):
            return "device"

    prepared = auto_matmul.PreparedMatmulInputs(
        input_tensor_a=FakeTensor(),
        input_tensor_b=FakeTensor(),
        bias=None,
    )
    calls = []

    monkeypatch.setattr(
        auto_matmul,
        "_ttnn",
        lambda: SimpleNamespace(
            Tensor=FakeTensor,
            CONFIG=SimpleNamespace(enable_graph_report=True),
        ),
    )
    monkeypatch.setattr(auto_matmul, "_prepare_inputs", lambda *args, **kwargs: prepared)
    monkeypatch.setattr(
        auto_matmul,
        "_select_candidate",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("selector should be bypassed")),
    )

    def base_operation(lhs, rhs, **kwargs):
        calls.append((lhs, rhs, kwargs))
        return "base-result"

    result = auto_matmul.dispatch_matmul(
        base_operation=base_operation,
        input_tensor_a=FakeTensor(),
        input_tensor_b=FakeTensor(),
        bias=None,
        is_linear=False,
        auto_config=True,
    )

    assert result == "base-result"
    assert calls == [(prepared.input_tensor_a, prepared.input_tensor_b, {})]


def test_dispatch_matmul_bypasses_selector_for_rank1_inputs(monkeypatch):
    class FakeTensor:
        shape = (4,)

        def device(self):
            return "device"

    a = FakeTensor()
    b = FakeTensor()
    calls = []

    monkeypatch.setattr(
        auto_matmul,
        "_ttnn",
        lambda: SimpleNamespace(
            Tensor=FakeTensor,
            CONFIG=SimpleNamespace(enable_graph_report=False),
        ),
    )
    monkeypatch.setattr(
        auto_matmul,
        "_select_candidate",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("selector should be bypassed")),
    )

    def base_operation(lhs, rhs, **kwargs):
        calls.append((lhs, rhs, kwargs))
        return "base-result"

    result = auto_matmul.dispatch_matmul(
        base_operation=base_operation,
        input_tensor_a=a,
        input_tensor_b=b,
        bias=None,
        is_linear=False,
        auto_config=True,
    )

    assert result == "base-result"
    assert calls == [(a, b, {})]


def test_explain_matmul_reports_unsupported_topology_passthrough(monkeypatch):
    class FakeTensor:
        def device(self):
            return "device"

    prepared = auto_matmul.PreparedMatmulInputs(
        input_tensor_a=FakeTensor(),
        input_tensor_b=FakeTensor(),
        bias=None,
    )
    signature = _make_distributed_signature(lhs_shard_dim=2)

    monkeypatch.setattr(auto_matmul, "_ttnn", lambda: SimpleNamespace(Tensor=FakeTensor))
    monkeypatch.setattr(auto_matmul, "_prepare_inputs", lambda *args, **kwargs: prepared)
    monkeypatch.setattr(auto_matmul, "_build_signature", lambda *args, **kwargs: signature)

    result = auto_matmul.explain_matmul(FakeTensor(), FakeTensor(), allow_tuning=True)

    assert result["winner"] == {"kind": "unsupported_topology_passthrough"}
    assert result["candidate_timings_us"] == []
    assert result["distributed_plan"]["kind"] == "unsupported"
    assert any("bypassed" in recommendation for recommendation in result["recommendations"])


# ---------------------------------------------------------------------------
# place_weight (host-weight distribution) helpers
# ---------------------------------------------------------------------------


def test_infer_distributed_plan_both_operands_sharded_on_k_is_reduce_scatter():
    # Both the activation (dim 3) and the weight (dim 2) sharded on their K dims:
    # this is what place_weight produces when it shards a host weight to match a
    # K-sharded activation, and it must route to reduce-scatter (not all-gather).
    signature = _make_distributed_signature(lhs_shard_dim=3, rhs_shard_dim=2)
    plan = auto_matmul._infer_distributed_plan(signature)
    assert plan.kind == "matmul_before_reduce_scatter"
    assert plan.cluster_axis == 0
    assert plan.lhs_shard_dim == 3
    assert plan.rhs_shard_dim == 2


def test_weight_contraction_dim_respects_transpose_b():
    assert auto_matmul._weight_contraction_dim((64, 128), transpose_b=False) == 0
    assert auto_matmul._weight_contraction_dim((128, 64), transpose_b=True) == 1
    assert auto_matmul._weight_contraction_dim((1, 1, 64, 128), transpose_b=False) == 2


def test_winner_avg_us_reads_fastest_ok_entry():
    selection = {
        "winner": {"kind": "shard"},
        "candidate_timings_us": [
            {"descriptor": {"kind": "a"}, "status": "incorrect"},
            {"descriptor": {"kind": "shard"}, "status": "ok", "average_us": 12.0},
            {"descriptor": {"kind": "b"}, "status": "error"},
        ],
    }
    assert auto_matmul._winner_avg_us(selection) == 12.0
    assert auto_matmul._winner_avg_us({"winner": None, "candidate_timings_us": []}) is None


def test_weight_placement_summary_round_trips_fields():
    placement = auto_matmul.WeightPlacement(
        tensor="t",
        strategy="shard_k",
        shard_dim=0,
        cluster_axis=0,
        mesh_shape=(1, 2),
        output_is_sharded=True,
        verified=True,
        candidate_timings=[{"strategy": "shard_k"}],
        recommendations=["r"],
    )
    summary = placement.summary()
    assert summary["strategy"] == "shard_k"
    assert summary["output_is_sharded"] is True
    assert summary["verified"] is True
    assert summary["mesh_shape"] == [1, 2]
    assert "tensor" not in summary


def _fake_ttnn_for_placement():
    class Tensor:
        pass

    return SimpleNamespace(
        Tensor=Tensor,
        TILE_LAYOUT="TILE",
        DRAM_MEMORY_CONFIG="DRAM",
        bfloat16="bf16",
        ReplicateTensorToMesh=lambda device: ("replicate", device),
        ShardTensorToMesh=lambda device, dim: ("shard", dim),
        ShardTensor2dMesh=lambda device, mesh_shape, dims: ("shard2d", dims),
    )


class _FakeMeshDevice:
    def __init__(self, num_devices, shape):
        self._num_devices = num_devices
        self.shape = shape

    def get_num_devices(self):
        return self._num_devices


def test_place_weight_passes_through_already_placed_tensor(monkeypatch):
    fake_ttnn = _fake_ttnn_for_placement()
    monkeypatch.setattr(auto_matmul, "_ttnn", lambda: fake_ttnn)
    already = fake_ttnn.Tensor()

    placement = auto_matmul.place_weight(already, mesh_device=_FakeMeshDevice(2, (1, 2)))

    assert placement.tensor is already
    assert placement.strategy == "preplaced"


def test_place_weight_single_device_replicates_without_measuring(monkeypatch):
    fake_ttnn = _fake_ttnn_for_placement()
    monkeypatch.setattr(auto_matmul, "_ttnn", lambda: fake_ttnn)
    staged = {}
    monkeypatch.setattr(
        auto_matmul, "_stage_weight", lambda weight, device, **kwargs: staged.setdefault(kwargs["role"], "staged")
    )
    weight = SimpleNamespace(shape=(64, 128))

    placement = auto_matmul.place_weight(weight, mesh_device=_FakeMeshDevice(1, (1,)))

    assert placement.strategy == "replicate"
    assert placement.output_is_sharded is False
    assert "weight_replicate" in staged


def test_place_weight_recommends_sharding_activation_when_not_k_sharded(monkeypatch):
    fake_ttnn = _fake_ttnn_for_placement()
    monkeypatch.setattr(auto_matmul, "_ttnn", lambda: fake_ttnn)
    monkeypatch.setattr(auto_matmul, "_activation_k_shard", lambda activation, transpose_a: (None, 1, 1))
    monkeypatch.setattr(auto_matmul, "_stage_weight", lambda weight, device, **kwargs: "staged")
    weight = SimpleNamespace(shape=(64, 128))

    placement = auto_matmul.place_weight(weight, mesh_device=_FakeMeshDevice(2, (1, 2)))

    assert placement.strategy == "replicate"
    assert placement.output_is_sharded is False
    assert any("row-parallel" in recommendation for recommendation in placement.recommendations)


def test_place_weight_shards_on_k_and_verifies_for_k_sharded_activation(monkeypatch):
    fake_ttnn = _fake_ttnn_for_placement()
    monkeypatch.setattr(auto_matmul, "_ttnn", lambda: fake_ttnn)
    activation = fake_ttnn.Tensor()
    monkeypatch.setattr(auto_matmul, "_activation_k_shard", lambda a, transpose_a: (0, 2, 1))
    monkeypatch.setattr(auto_matmul, "_stage_weight", lambda weight, device, **kwargs: kwargs["role"])
    monkeypatch.setattr(auto_matmul, "_get_cpp_base_operation", lambda is_linear: "base")
    monkeypatch.setattr(auto_matmul, "_placement_reference_output", lambda *args, **kwargs: "reference")

    def fake_measure(activation, staged, **kwargs):
        assert staged == "weight_shard_k"
        return {"avg_us": 8.0, "winner": {"kind": "reduce_scatter"}, "verified": True, "output_is_sharded": True}

    monkeypatch.setattr(auto_matmul, "_measure_weight_placement", fake_measure)
    weight = SimpleNamespace(shape=(64, 128))

    placement = auto_matmul.place_weight(
        weight, activation=activation, mesh_device=_FakeMeshDevice(2, (2,)), measure=True
    )

    assert placement.strategy == "shard_k"
    assert placement.tensor == "weight_shard_k"
    assert placement.output_is_sharded is True
    assert placement.verified is True
    assert placement.cluster_axis == 0


def test_place_weight_does_not_execute_a_collective_by_default(monkeypatch):
    # Measurement runs the reduce-scatter collective, which can hang on non-ring
    # hardware; place_weight must NOT do that unless measure=True is passed.
    fake_ttnn = _fake_ttnn_for_placement()
    monkeypatch.setattr(auto_matmul, "_ttnn", lambda: fake_ttnn)
    activation = fake_ttnn.Tensor()
    monkeypatch.setattr(auto_matmul, "_activation_k_shard", lambda a, transpose_a: (0, 2, 1))
    monkeypatch.setattr(auto_matmul, "_stage_weight", lambda weight, device, **kwargs: kwargs["role"])

    def explode(*args, **kwargs):
        raise AssertionError("place_weight must not measure/execute a collective by default")

    monkeypatch.setattr(auto_matmul, "_measure_weight_placement", explode)
    weight = SimpleNamespace(shape=(64, 128))

    placement = auto_matmul.place_weight(weight, activation=activation, mesh_device=_FakeMeshDevice(2, (2,)))

    assert placement.strategy == "shard_k"
    assert placement.verified is False
    assert any("not executed/verified" in rec for rec in placement.recommendations)


def test_place_weight_reports_unverified_shard_when_ground_truth_fails(monkeypatch):
    fake_ttnn = _fake_ttnn_for_placement()
    monkeypatch.setattr(auto_matmul, "_ttnn", lambda: fake_ttnn)
    activation = fake_ttnn.Tensor()
    monkeypatch.setattr(auto_matmul, "_activation_k_shard", lambda a, transpose_a: (0, 2, 1))
    monkeypatch.setattr(auto_matmul, "_stage_weight", lambda weight, device, **kwargs: kwargs["role"])
    monkeypatch.setattr(auto_matmul, "_get_cpp_base_operation", lambda is_linear: "base")
    monkeypatch.setattr(auto_matmul, "_placement_reference_output", lambda *args, **kwargs: "reference")

    def fake_measure(activation, staged, **kwargs):
        return {"avg_us": None, "winner": None, "verified": False, "output_is_sharded": True}

    monkeypatch.setattr(auto_matmul, "_measure_weight_placement", fake_measure)
    weight = SimpleNamespace(shape=(64, 128))

    placement = auto_matmul.place_weight(
        weight, activation=activation, mesh_device=_FakeMeshDevice(2, (2,)), measure=True
    )

    # The K-sharded weight is the only shape-consistent placement for a K-sharded
    # activation, so it is returned but flagged unverified with a recommendation.
    assert placement.strategy == "shard_k"
    assert placement.verified is False
    assert any("did not match the ground-truth" in rec for rec in placement.recommendations)


# ---------------------------------------------------------------------------
# Device-profiler based benchmarking (device kernel duration, no dispatch overhead)
# ---------------------------------------------------------------------------


def _profiler_env(monkeypatch, enabled=True):
    for flag in auto_matmul._PROFILER_ENV_FLAGS:
        if enabled:
            monkeypatch.setenv(flag, "1")
        else:
            monkeypatch.delenv(flag, raising=False)


def _ttnn_with_profiler(**overrides):
    profiler_ns = SimpleNamespace()
    ns = SimpleNamespace(
        _ttnn=SimpleNamespace(profiler=profiler_ns),
        get_latest_programs_perf_data=lambda: {},
        ReadDeviceProfiler=lambda device: None,
    )
    for key, value in overrides.items():
        setattr(ns, key, value)
    return ns


def test_device_profiler_enabled_requires_bindings_and_env(monkeypatch):
    _profiler_env(monkeypatch, enabled=True)
    monkeypatch.setattr(auto_matmul, "_ttnn", lambda: _ttnn_with_profiler())
    assert auto_matmul._device_profiler_enabled() is True

    # Missing env flags -> disabled.
    _profiler_env(monkeypatch, enabled=False)
    assert auto_matmul._device_profiler_enabled() is False

    # Env set but no profiler bindings in the build -> disabled.
    _profiler_env(monkeypatch, enabled=True)
    monkeypatch.setattr(auto_matmul, "_ttnn", lambda: SimpleNamespace(_ttnn=SimpleNamespace()))
    assert auto_matmul._device_profiler_enabled() is False


def _perf_program(duration_ns):
    result = SimpleNamespace(duration=duration_ns)
    return SimpleNamespace(program_analyses_results={auto_matmul._DEVICE_KERNEL_DURATION_KEY: result})


def test_latest_device_kernel_duration_sums_programs_and_takes_slowest_chip(monkeypatch):
    # Chip 0 runs two programs (500 + 700 ns); chip 1 runs one (900 ns).
    # Per-chip totals: {0: 1200, 1: 900}; the critical path is the slowest chip.
    perf_data = {
        0: [_perf_program(500), _perf_program(700)],
        1: [_perf_program(900)],
    }
    monkeypatch.setattr(
        auto_matmul, "_ttnn", lambda: _ttnn_with_profiler(get_latest_programs_perf_data=lambda: perf_data)
    )
    assert auto_matmul._latest_device_kernel_duration_ns("dev") == 1200.0


def test_latest_device_kernel_duration_returns_none_when_empty(monkeypatch):
    monkeypatch.setattr(auto_matmul, "_ttnn", lambda: _ttnn_with_profiler(get_latest_programs_perf_data=lambda: {}))
    assert auto_matmul._latest_device_kernel_duration_ns("dev") is None


def test_benchmark_candidate_profiler_reports_device_kernel_duration_in_us(monkeypatch):
    # Warmup read + one read per iteration; return 2000 ns each iteration -> 2.0 us.
    perf_data = {0: [_perf_program(2000)]}
    monkeypatch.setattr(
        auto_matmul, "_ttnn", lambda: _ttnn_with_profiler(get_latest_programs_perf_data=lambda: perf_data)
    )
    monkeypatch.setattr(auto_matmul, "_sync_device", lambda device: None)
    monkeypatch.setattr(auto_matmul, "_deallocate_result", lambda result: None)

    candidate = auto_matmul.Candidate(descriptor={"kind": "minimal_matmul"}, run=lambda: "out")
    avg_us, samples_us, mode = auto_matmul._benchmark_candidate_profiler(candidate, "dev")

    assert mode == "profiler"
    assert avg_us == pytest.approx(2.0)
    assert samples_us == pytest.approx([2.0] * auto_matmul._BENCHMARK_ITERS)


def test_benchmark_candidate_prefers_profiler_when_enabled(monkeypatch):
    _profiler_env(monkeypatch, enabled=True)
    perf_data = {0: [_perf_program(3000)]}
    monkeypatch.setattr(
        auto_matmul, "_ttnn", lambda: _ttnn_with_profiler(get_latest_programs_perf_data=lambda: perf_data)
    )
    monkeypatch.setattr(auto_matmul, "_sync_device", lambda device: None)
    monkeypatch.setattr(auto_matmul, "_deallocate_result", lambda result: None)

    def _fail(*args, **kwargs):
        raise AssertionError("wall-clock fallback should not run when the profiler is active")

    monkeypatch.setattr(auto_matmul, "_benchmark_candidate_trace", _fail)
    monkeypatch.setattr(auto_matmul, "_benchmark_candidate_eager", _fail)

    candidate = auto_matmul.Candidate(descriptor={"kind": "minimal_matmul"}, run=lambda: "out")
    avg_us, _samples, mode = auto_matmul._benchmark_candidate(candidate, "dev")
    assert mode == "profiler"
    assert avg_us == pytest.approx(3.0)


def test_benchmark_candidate_falls_back_to_wall_clock_without_profiler(monkeypatch):
    _profiler_env(monkeypatch, enabled=False)
    monkeypatch.setattr(
        auto_matmul,
        "_ttnn",
        lambda: SimpleNamespace(begin_trace_capture=1, end_trace_capture=1, execute_trace=1, release_trace=1),
    )
    monkeypatch.setattr(
        auto_matmul,
        "_benchmark_candidate_trace",
        lambda candidate, device, queue_id=0: (5.0, [5.0], "trace"),
    )
    candidate = auto_matmul.Candidate(descriptor={"kind": "minimal_matmul"}, run=lambda: "out")
    avg_us, _samples, mode = auto_matmul._benchmark_candidate(candidate, "dev")
    assert mode == "trace"
    assert avg_us == pytest.approx(5.0)
