# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Bounded silicon canary for a populated one-chip Blackhole matmul registry.

The launcher runs this module in a fresh process for Off, Shadow, and On because
the registry mode is intentionally frozen on first public matmul dispatch.
Ordinary device test runs skip an empty/inapplicable table; the release canary
sets ``TTNN_MATMUL_REGISTRY_CANARY_REQUIRE_POPULATED=1`` and fails closed.
"""

from __future__ import annotations

import json
import os
import struct
from pathlib import Path
from typing import Any

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc

pytestmark = pytest.mark.use_module_device

ROOT = Path(__file__).resolve().parents[5]
LOCK_PATH = ROOT / "ttnn/cpp/ttnn/operations/matmul/device/config/registry/matmul_registry.lock.json"
MODE_NAME = os.environ.get("TTNN_MATMUL_REGISTRY_CANARY_MODE")
REQUIRE_POPULATED = os.environ.get("TTNN_MATMUL_REGISTRY_CANARY_REQUIRE_POPULATED") == "1"
MAX_INPUT_ELEMENTS = int(os.environ.get("TTNN_MATMUL_REGISTRY_CANARY_MAX_INPUT_ELEMENTS", "16000000"))
MODE_VALUE = {"off": 0, "shadow": 1, "on": 2}
REGISTRY = ttnn._ttnn.operations.matmul


def _skip_or_fail(message: str) -> None:
    if REQUIRE_POPULATED:
        pytest.fail(message)
    pytest.skip(message)


def _lock() -> dict[str, Any]:
    value = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    if value.get("artifact_kind") != "ttnn_matmul_registry_lock" or value.get("lock_schema_version") != 1:
        pytest.fail(f"unsupported registry lock at {LOCK_PATH}")
    if not value.get("entries"):
        _skip_or_fail("matmul registry lock is empty; populated-lock silicon canary did not run")
    return value


def _float32_bits(value: float) -> int:
    return struct.unpack("!I", struct.pack("!f", value))[0]


def _plain(value: Any) -> Any:
    if hasattr(value, "keys"):
        return {str(key): _plain(value[key]) for key in value.keys()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _domain_stats(snapshot: dict[str, Any], domain: str) -> dict[str, Any]:
    matches = [item for item in snapshot["domains"] if item["domain"] == domain]
    assert len(matches) == 1
    return _plain(matches[0])


def _delta(before: dict[str, Any], after: dict[str, Any], name: str) -> int:
    return int(after[name]) - int(before[name])


def _assert_reason_delta(before: dict[str, Any], after: dict[str, Any], reason: str, expected: int) -> None:
    before_reasons = before["reason_counts_by_name"]
    after_reasons = after["reason_counts_by_name"]
    assert set(before_reasons) == set(after_reasons)
    assert int(after_reasons[reason]) - int(before_reasons[reason]) == expected


def _compatible_entries(lock: dict[str, Any], device, domain: str, *, beta_bits: int | None = None) -> list[dict]:
    attestation = _plain(REGISTRY.matmul_registry_compatibility_attestation(device))
    if attestation["device_attestation_status"] != "success":
        _skip_or_fail(f"registry device attestation failed: {attestation['device_attestation_status']}")
    for lock_name, attestation_name in (
        ("semantic_source_sha256", "actual_semantic_source_sha256"),
        ("build_identity_sha256", "actual_build_identity_sha256"),
        ("runtime_capability_sha256", "actual_runtime_capability_sha256"),
    ):
        if lock[lock_name] != attestation[attestation_name]:
            _skip_or_fail(f"checked lock and running binary disagree on {lock_name}")

    entries = []
    for entry in lock["entries"]:
        key = entry["key"]
        if entry["domain"] != domain:
            continue
        if domain == "dense.addmm" and key["alpha_f32_bits"] != _float32_bits(1.0):
            continue
        if beta_bits is not None and key["beta_f32_bits"] != beta_bits:
            continue
        if key["topology_sha256"] != attestation["actual_topology_sha256"]:
            continue
        if key["board_capability_class"] != attestation["board_capability_class"]:
            continue
        cost = key["logical_m"] * key["logical_k"] + key["logical_k"] * key["logical_n"]
        if cost <= MAX_INPUT_ELEMENTS:
            entries.append(entry)
    entries.sort(
        key=lambda entry: (
            entry["key"]["logical_m"] * entry["key"]["logical_k"]
            + entry["key"]["logical_k"] * entry["key"]["logical_n"],
            entry["entry_id"],
        )
    )
    if not entries:
        suffix = f" beta_bits={beta_bits:#x}" if beta_bits is not None else ""
        _skip_or_fail(f"no bounded, topology-compatible {domain}{suffix} registry entry")
    return entries


def _tt_dtype(name: str):
    return {"bfloat16": ttnn.bfloat16, "bfloat8_b": ttnn.bfloat8_b, "float32": ttnn.float32}[name]


def _memory_config(name: str):
    return {"dram": ttnn.DRAM_MEMORY_CONFIG, "l1": ttnn.L1_MEMORY_CONFIG}[name]


def _make_tensor(host: torch.Tensor, descriptor: dict[str, Any], device):
    assert descriptor["layout"] == "tile"
    assert descriptor["memory_layout"] == "interleaved"
    return ttnn.from_torch(
        host,
        dtype=_tt_dtype(descriptor["dtype"]),
        layout=ttnn.TILE_LAYOUT,
        tile=ttnn.Tile((descriptor["tile_height"], descriptor["tile_width"])),
        memory_config=_memory_config(descriptor["buffer_type"]),
        device=device,
    )


def _inputs(entry: dict[str, Any], device, *, logical_m: int | None = None):
    key = entry["key"]
    m = logical_m if logical_m is not None else key["logical_m"]
    generator = torch.Generator().manual_seed(int(entry["entry_id"][:16], 16))
    host_a = torch.randn((m, key["logical_k"]), generator=generator, dtype=torch.float32)
    host_b = torch.randn((key["logical_k"], key["logical_n"]), generator=generator, dtype=torch.float32)
    tensor_a = _make_tensor(host_a, key["input_a"], device)
    tensor_b = _make_tensor(host_b, key["input_b"], device)
    return host_a, host_b, tensor_a, tensor_b


def _resolved_key(tensor_a, tensor_b, domain: str, output: dict[str, Any], *, beta: float | None = None) -> dict:
    kwargs = {
        "domain": domain,
        "memory_config": _memory_config(output["buffer_type"]),
        "dtype": _tt_dtype(output["dtype"]),
    }
    if domain == "dense.addmm":
        kwargs.update(alpha=1.0, beta=beta)
    report = _plain(REGISTRY.matmul_registry_resolved_key(tensor_a, tensor_b, **kwargs))
    assert report["device_attestation_status"] == "success"
    return report["native_registry_key_v1"]


def _invoke(
    domain: str,
    entry: dict[str, Any],
    host_a: torch.Tensor,
    host_b: torch.Tensor,
    tensor_a,
    tensor_b,
    *,
    beta: float | None = None,
    bias=None,
    output_tile=None,
):
    key = entry["key"]
    kwargs = {
        "memory_config": _memory_config(key["output"]["buffer_type"]),
        "dtype": _tt_dtype(key["output"]["dtype"]),
    }
    expected = host_a @ host_b
    if domain == "dense.matmul":
        output = ttnn.matmul(tensor_a, tensor_b, output_tile=output_tile, **kwargs)
    elif domain == "dense.linear":
        output = ttnn.linear(tensor_a, tensor_b, bias=bias, **kwargs)
        if bias is not None:
            expected = expected + ttnn.to_torch(bias).to(torch.float32)
    else:
        additive = torch.randn(expected.shape, generator=torch.Generator().manual_seed(7), dtype=torch.float32)
        additive_tensor = ttnn.from_torch(
            additive, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=tensor_a.device()
        )
        output = ttnn.addmm(additive_tensor, tensor_a, tensor_b, alpha=1.0, beta=beta, **kwargs)
        expected = expected + float(beta) * additive
    actual = ttnn.to_torch(output).to(torch.float32)
    threshold = 0.995 if "bfloat8" in (key["input_a"]["dtype"], key["input_b"]["dtype"]) else 0.999
    assert_with_pcc(expected, actual, pcc=threshold)
    return output


def _assert_dispatch_delta(before: dict, after: dict, *, exact_hit: bool, reason: str | None = None) -> None:
    mode = MODE_NAME
    assert mode in MODE_VALUE
    if mode == "off":
        for field in (
            "resolution_attempts",
            "certified_hits",
            "shadow_would_hits",
            "selected_hits",
            "completed_hits",
            "fallbacks",
            "circuit_breaker_activations",
        ):
            assert _delta(before, after, field) == 0
        assert before["reason_counts_by_name"] == after["reason_counts_by_name"]
        assert not after["circuit_broken"]
        return

    assert _delta(before, after, "resolution_attempts") == 1
    assert _delta(before, after, "circuit_breaker_activations") == 0
    assert not after["circuit_broken"]
    if exact_hit:
        _assert_reason_delta(before, after, "certified_match", 1)
        assert _delta(before, after, "certified_hits") == 1
        assert _delta(before, after, "fallbacks") == 0
        if mode == "shadow":
            assert _delta(before, after, "shadow_would_hits") == 1
            assert _delta(before, after, "selected_hits") == 0
            assert _delta(before, after, "completed_hits") == 0
        else:
            assert _delta(before, after, "shadow_would_hits") == 0
            assert _delta(before, after, "selected_hits") == 1
            assert _delta(before, after, "completed_hits") == 1
    else:
        assert reason is not None
        _assert_reason_delta(before, after, reason, 1)
        assert _delta(before, after, "certified_hits") == 0
        assert _delta(before, after, "shadow_would_hits") == 0
        assert _delta(before, after, "selected_hits") == 0
        assert _delta(before, after, "completed_hits") == 0
        assert _delta(before, after, "fallbacks") == 1


@pytest.fixture(scope="module", autouse=True)
def validate_process_contract(device):
    if MODE_NAME not in MODE_VALUE:
        pytest.skip("run through run_bh_matmul_registry_canary.sh so registry mode is process-frozen")
    lock = _lock()
    stats = _plain(REGISTRY.matmul_registry_stats())
    assert stats["entry_count"] == len(lock["entries"]), "running binary does not contain the checked lock"
    assert int(ttnn.CONFIG.matmul_registry_mode) == MODE_VALUE[MODE_NAME]
    yield
    final = _plain(REGISTRY.matmul_registry_stats())
    assert final["mode_is_frozen"]
    assert final["frozen_mode"] == MODE_VALUE[MODE_NAME]


@pytest.mark.parametrize(
    "domain,beta_bits,beta",
    [
        ("dense.matmul", None, None),
        ("dense.linear", None, None),
        ("dense.addmm", _float32_bits(0.0), 0.0),
        ("dense.addmm", _float32_bits(-0.0), -0.0),
    ],
)
def test_exact_public_call_uses_expected_registry_mode(device, domain, beta_bits, beta):
    lock = _lock()
    entry = _compatible_entries(lock, device, domain, beta_bits=beta_bits)[0]
    host_a, host_b, tensor_a, tensor_b = _inputs(entry, device)
    assert _resolved_key(tensor_a, tensor_b, domain, entry["key"]["output"], beta=beta) == {
        "domain": domain,
        "key": entry["key"],
    }

    before = _domain_stats(_plain(REGISTRY.matmul_registry_stats()), domain)
    _invoke(domain, entry, host_a, host_b, tensor_a, tensor_b, beta=beta)
    after = _domain_stats(_plain(REGISTRY.matmul_registry_stats()), domain)
    _assert_dispatch_delta(before, after, exact_hit=True)


def test_unsupported_public_variants_fall_back_without_circuit_break(device):
    lock = _lock()
    cases = (
        ("dense.matmul", "output_tile", "unsupported_semantics"),
        ("dense.linear", "bias", "unsupported_semantics"),
        ("dense.addmm", "beta", "unsupported_semantics"),
    )
    for domain, variant, reason in cases:
        entry = _compatible_entries(
            lock, device, domain, beta_bits=_float32_bits(0.0) if domain == "dense.addmm" else None
        )[0]
        host_a, host_b, tensor_a, tensor_b = _inputs(entry, device)
        bias = None
        output_tile = None
        beta = 0.0
        if variant == "bias":
            host_bias = torch.randn((1, entry["key"]["logical_n"]), dtype=torch.float32)
            bias = ttnn.from_torch(host_bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        elif variant == "output_tile":
            output_tile = ttnn.Tile((32, 32))
        else:
            beta = 1.0
        before = _domain_stats(_plain(REGISTRY.matmul_registry_stats()), domain)
        _invoke(
            domain,
            entry,
            host_a,
            host_b,
            tensor_a,
            tensor_b,
            beta=beta,
            bias=bias,
            output_tile=output_tile,
        )
        after = _domain_stats(_plain(REGISTRY.matmul_registry_stats()), domain)
        _assert_dispatch_delta(before, after, exact_hit=False, reason=reason)


def test_exact_shape_miss_falls_back(device):
    lock = _lock()
    domain = "dense.matmul"
    entry = _compatible_entries(lock, device, domain)[0]
    all_keys = {
        json.dumps({"domain": item["domain"], "key": item["key"]}, sort_keys=True, separators=(",", ":"))
        for item in lock["entries"]
    }
    miss = None
    for logical_m in (32, 64, 96, 128, 160, 192, 224, 256, 384, 512, 768, 1024):
        if logical_m * entry["key"]["logical_k"] > MAX_INPUT_ELEMENTS:
            continue
        host_a, host_b, tensor_a, tensor_b = _inputs(entry, device, logical_m=logical_m)
        native = _resolved_key(tensor_a, tensor_b, domain, entry["key"]["output"])
        if json.dumps(native, sort_keys=True, separators=(",", ":")) not in all_keys:
            miss = (host_a, host_b, tensor_a, tensor_b)
            break
    if miss is None:
        _skip_or_fail("could not construct a bounded exact-key miss")

    before = _domain_stats(_plain(REGISTRY.matmul_registry_stats()), domain)
    _invoke(domain, entry, *miss)
    after = _domain_stats(_plain(REGISTRY.matmul_registry_stats()), domain)
    _assert_dispatch_delta(before, after, exact_hit=False, reason="empty_registry")


def test_public_validation_error_is_not_retried(device, expect_error):
    lock = _lock()
    domain = "dense.matmul"
    entry = _compatible_entries(lock, device, domain)[0]
    key = entry["key"]
    generator = torch.Generator().manual_seed(11)
    host_a = torch.randn((key["logical_m"], key["logical_k"] + 32), generator=generator)
    host_b = torch.randn((key["logical_k"], key["logical_n"]), generator=generator)
    tensor_a = _make_tensor(host_a, key["input_a"], device)
    tensor_b = _make_tensor(host_b, key["input_b"], device)
    before = _domain_stats(_plain(REGISTRY.matmul_registry_stats()), domain)
    with expect_error(RuntimeError, "."):
        ttnn.matmul(
            tensor_a,
            tensor_b,
            memory_config=_memory_config(key["output"]["buffer_type"]),
            dtype=_tt_dtype(key["output"]["dtype"]),
        )
    after = _domain_stats(_plain(REGISTRY.matmul_registry_stats()), domain)
    _assert_dispatch_delta(before, after, exact_hit=False, reason="incomplete_request")
    assert _delta(before, after, "resolution_attempts") == (0 if MODE_NAME == "off" else 1)
