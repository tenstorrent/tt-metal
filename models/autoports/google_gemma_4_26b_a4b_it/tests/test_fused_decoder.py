# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Stage-02 tests for the Gemma4 graph-fused decoder."""

import inspect
import hashlib
from pathlib import Path

from models.autoports.google_gemma_4_26b_a4b_it.tt.fused_decoder import FusedDecoder


def test_fused_decoder_is_a_distinct_runtime_class():
    assert FusedDecoder.__name__ == "FusedDecoder"
    assert "_dense_mlp" in FusedDecoder.__dict__
    assert "_router_weights" in FusedDecoder.__dict__
    assert "_moe_decode_single_user" in FusedDecoder.__dict__
    assert "_moe_prefill_chunk" in FusedDecoder.__dict__
    assert Path(__import__(FusedDecoder.__module__, fromlist=["x"]).__file__).name == "fused_decoder.py"


def test_fused_dense_mlp_has_no_standalone_gelu(monkeypatch):
    calls = []

    class FakeTTNN:
        DRAM_MEMORY_CONFIG = object()

        class UnaryOpType:
            GELU = "gelu"

        @staticmethod
        def UnaryWithParam(op, parameter):
            return op, parameter

        @staticmethod
        def linear(x, weight, **kwargs):
            calls.append(("linear", x, weight, kwargs))
            return f"linear-{len(calls)}"

        @staticmethod
        def mul(a, b, **kwargs):
            calls.append(("mul", a, b, kwargs))
            return "fused-geglu"

    import models.autoports.google_gemma_4_26b_a4b_it.tt.fused_decoder as module

    monkeypatch.setattr(module, "ttnn", FakeTTNN)
    weights = type("Weights", (), {"mlp_gate": "gate", "mlp_up": "up", "mlp_down": "down"})()
    decoder = object.__new__(FusedDecoder)
    decoder.weights = weights
    decoder.activation_dtype = "bf16"

    assert decoder._dense_mlp("x") == "linear-4"
    assert [call[0] for call in calls] == ["linear", "linear", "mul", "linear"]
    assert calls[2][3]["input_tensor_a_activations"] == [("gelu", 1.0)]


def test_fused_router_scale_is_folded_at_setup(monkeypatch):
    calls = []

    class FakeTTNN:
        DRAM_MEMORY_CONFIG = "dram"

        @staticmethod
        def mul(a, b, **kwargs):
            calls.append((a, b, kwargs))
            return "folded-scale"

    import models.autoports.google_gemma_4_26b_a4b_it.tt.fused_decoder as module

    def fake_base_init(decoder, **_):
        decoder.weights = type("Weights", (), {"router_scale": "router-scale"})()
        decoder.router_hidden_scale = "hidden-scale"

    monkeypatch.setattr(module, "ttnn", FakeTTNN)
    monkeypatch.setattr(module.FunctionalDecoder, "__init__", fake_base_init)
    decoder = FusedDecoder()

    assert decoder.fused_router_scale == "folded-scale"
    assert calls == [("router-scale", "hidden-scale", {"memory_config": "dram"})]


def test_fused_hot_path_has_no_host_fallback():
    source = "\n".join(
        inspect.getsource(getattr(FusedDecoder, name))
        for name in (
            "_router_weights",
            "_dense_mlp",
            "_moe_decode_single_user",
            "_moe_prefill_chunk",
            "_moe_prefill_tile_group",
        )
    )
    for forbidden in (
        "torch.",
        "ttnn.from_torch",
        "ttnn.to_torch",
        "apply_geglu",
        "FunctionalDecoder._dense_mlp",
        "FunctionalDecoder._moe_prefill_chunk",
    ):
        assert forbidden not in source
    assert source.count("input_tensor_a_activations") == 3


def test_inherited_prefill_dispatches_to_fused_chunk(monkeypatch):
    calls = []
    decoder = object.__new__(FusedDecoder)
    decoder._moe_prefill_chunk = lambda hidden, routing: calls.append((hidden, routing)) or "fused-prefill"
    hidden = type("Tensor", (), {"shape": (1, 1, 32, 2816)})()

    assert decoder._moe_prefill(hidden, "routing") == "fused-prefill"
    assert calls == [(hidden, "routing")]


def test_prefill_tile_group_and_router_contracts_are_structurally_fused():
    chunk_source = inspect.getsource(FusedDecoder._moe_prefill_chunk)
    tile_source = inspect.getsource(FusedDecoder._moe_prefill_tile_group)
    router_source = inspect.getsource(FusedDecoder._router_weights)

    assert "ttnn.split(hidden_states, TILE_SIZE, dim=2)" in chunk_source
    assert "ttnn.split(routing_weights, TILE_SIZE, dim=2)" in chunk_source
    assert "ttnn.concat([result, chunk_result], dim=2)" in chunk_source
    assert tile_source.count("ttnn.sparse_matmul(") == 3
    assert "nnz = NUM_EXPERTS * group_size" in tile_source
    assert "input_tensor_a_activations" in tile_source
    assert "ttnn.to_layout(down_input, ttnn.TILE_LAYOUT)" in tile_source
    assert "self.fused_router_scale" in router_source
    assert "self.router_hidden_scale" not in router_source


def test_final_manifest_is_complete_and_fail_closed():
    repo = Path(__file__).resolve().parents[4]
    evidence = repo / "models/autoports/google_gemma_4_26b_a4b_it/doc/fused_decoder"
    checksum_file = evidence / "final_manifest.sha256"
    entries = {}
    for line in checksum_file.read_text().splitlines():
        digest, relative_path = line.split(maxsplit=1)
        assert relative_path not in entries
        entries[relative_path] = digest

    required = {
        str(evidence.relative_to(repo) / "bounded_modulo_tail_cache_integrity.json"),
        str(evidence.relative_to(repo) / "profiler_summary.md"),
        str(evidence.relative_to(repo) / "tracy/final_ops_sliding_b1/prefill_report.csv"),
        str(evidence.relative_to(repo) / "tracy/final_ops_sliding_b1/prefill_report.txt"),
        str(evidence.relative_to(repo) / "tracy/final_ops_full_b1/prefill_report.csv"),
        str(evidence.relative_to(repo) / "tracy/final_ops_full_b1/prefill_report.txt"),
        str(evidence.relative_to(repo) / "tracy/final_sliding_b1/.logs/cpp_device_perf_report.csv"),
        str(evidence.relative_to(repo) / "tracy/final_full_b1/.logs/cpp_device_perf_report.csv"),
        str(evidence.relative_to(repo) / "tracy/final_sliding_b32/.logs/cpp_device_perf_report.csv"),
        str(evidence.relative_to(repo) / "tracy/final_full_b32/.logs/cpp_device_perf_report.csv"),
    }
    assert required <= entries.keys()

    # Derive the complete accepted fused-stage evidence set from the gate
    # artifact families, rather than allowing the hand-written minimum above
    # to silently omit a new result.
    accepted_files = {
        path.relative_to(repo).as_posix()
        for pattern in (
            "advertised_context_decode_*.json",
            "bounded_modulo_tail_cache_integrity.json",
            "layer*_host_timings.json",
            "pcc_layer*.json",
            "prefill_batch2_*.json",
            "prefill_boundaries_*.json",
            "trace_*.json",
            "watcher.log",
            "profiler_summary.md",
            "tracy/final_ops_*_b1/prefill_report.csv",
            "tracy/final_ops_*_b1/prefill_report.txt",
            "tracy/final_ops_*_b1.log",
            "tracy/final_ops_*_b1_legacy.log",
            "tracy/final_*.log",
        )
        for path in evidence.glob(pattern)
    }
    accepted_files.update(
        (evidence / f"tracy/final_{layer}_b{batch}/.logs/cpp_device_perf_report.csv")
        .relative_to(repo)
        .as_posix()
        for layer in ("sliding", "full")
        for batch in (1, 32)
    )
    accepted_files.add(
        (evidence / "autofix_functional_full_b1_201.json").relative_to(repo).as_posix()
    )
    accepted_files.update(
        path.relative_to(repo).as_posix()
        for path in (
            repo / "models/autoports/google_gemma_4_26b_a4b_it/tt/fused_decoder.py",
            repo / "models/autoports/google_gemma_4_26b_a4b_it/tt/functional_decoder.py",
            repo / "models/autoports/google_gemma_4_26b_a4b_it/tests/test_fused_decoder.py",
            repo / "models/autoports/google_gemma_4_26b_a4b_it/tests/test_functional_decoder.py",
            repo / "ttnn/ttnn/_ttnn.so",
        )
    )
    accepted_files.update(
        path.relative_to(repo).as_posix()
        for path in (
            repo
            / "models/autoports/google_gemma_4_26b_a4b_it/doc/functional_decoder"
        ).glob("layer*_host_timings.json")
    )
    assert accepted_files == entries.keys()

    for relative_path, expected in entries.items():
        path = repo / relative_path
        assert path.is_file(), relative_path
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, relative_path
