# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import dataclasses
from types import SimpleNamespace

import torch

from ttnn._experimental.auto_config import matmul as auto_matmul


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
            "memory_config": "DRAM_MEMORY_CONFIG",
            "topology": None,
        },
        input_tensor_b={
            "shape": [96, 128],
            "dtype": "bfloat16",
            "layout": "Layout.TILE",
            "memory_config": "DRAM_MEMORY_CONFIG",
            "topology": None,
        },
        bias=None,
        m=64,
        k=96,
        n=128,
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
            "memory_config": "DRAM_MEMORY_CONFIG",
            "topology": lhs_topology,
        },
        input_tensor_b={
            "shape": [1, 1, 64, 128],
            "dtype": "bfloat16",
            "layout": "Layout.TILE",
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
    assert auto_matmul._pick_subblock(3, 5) == (1, 1)


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
        def device(self):
            return "device"

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
