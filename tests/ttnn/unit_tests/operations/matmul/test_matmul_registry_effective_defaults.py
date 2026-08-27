# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re

import pytest
import torch

import ttnn

pytestmark = pytest.mark.use_module_device

REGISTRY = ttnn._ttnn.operations.matmul
SHA256 = re.compile(r"[0-9a-f]{64}")


def _plain(value):
    if isinstance(value, dict):
        return {str(key): _plain(value[key]) for key in value}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _inputs(device):
    tensor_a = ttnn.from_torch(
        torch.zeros((128, 256), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        device=device,
    )
    tensor_b = ttnn.from_torch(
        torch.zeros((256, 512), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        device=device,
    )
    return tensor_a, tensor_b


def _compact_program():
    return ttnn.MatmulMultiCoreReuseProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
        in0_block_w=2,
        out_subblock_h=1,
        out_subblock_w=2,
        per_core_M=4,
        per_core_N=16,
    )


def _report(device) -> dict:
    tensor_a, tensor_b = _inputs(device)
    return _plain(
        REGISTRY.matmul_registry_effective_default_compute_kernel_config(
            tensor_a,
            tensor_b,
            program_config=_compact_program(),
            domain="dense.matmul",
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
        )
    )


def test_effective_default_ckc_is_bound_to_exact_context_and_attestation(device) -> None:
    before = _plain(REGISTRY.matmul_registry_stats())
    report = _report(device)
    after = _plain(REGISTRY.matmul_registry_stats())

    assert report["artifact_kind"] == "ttnn_matmul_effective_default_compute_kernel_config"
    assert report["schema_version"] == 1
    assert report["device_attestation_status"] == "success"
    assert type(report["codegen_recipe_abi"]) is int and report["codegen_recipe_abi"] > 0
    assert type(report["board_capability_class"]) is int
    for name in (
        "actual_semantic_source_sha256",
        "actual_build_identity_sha256",
        "actual_topology_sha256",
        "actual_runtime_capability_sha256",
    ):
        assert SHA256.fullmatch(report[name])
        assert report[name] != "0" * 64

    context = report["context"]
    native = context["native_registry_key_v1"]
    assert native["domain"] == "dense.matmul"
    assert native["key"]["logical_m"] == 128
    assert native["key"]["logical_k"] == 256
    assert native["key"]["logical_n"] == 512
    assert native["key"]["topology_sha256"] == report["actual_topology_sha256"]
    assert native["key"]["board_capability_class"] == report["board_capability_class"]
    assert native["key"]["codegen_recipe_abi"] == report["codegen_recipe_abi"]
    assert context["caller_compute_kernel_config"] is None
    assert context["caller_core_grid"] is None
    assert context["program_config"] == {
        "allowed_worker_cores": None,
        "compute_grid_x": 8,
        "compute_grid_y": 8,
        "family": "multi_core_reuse",
        "in0_block_w": 2,
        "out_subblock_h": 1,
        "out_subblock_w": 2,
        "per_core_m": 4,
        "per_core_n": 16,
    }
    assert context["admitted_call_state_v1"]["output"] == native["key"]["output"]
    assert report["effective_compute_kernel_config"] == {
        "dst_full_sync_en": False,
        "fp32_dest_acc_en": False,
        "math_approx_mode": False,
        "math_fidelity": "lofi",
        "packer_l1_acc": True,
        "throttle_level": "no_throttle",
    }

    # Read-only evidence collection cannot initialize compatibility, resolve a
    # selector, or mutate bounded registry telemetry.
    assert after == before


def test_effective_default_ckc_rejects_caller_overrides_and_unsupported_family(device, expect_error) -> None:
    tensor_a, tensor_b = _inputs(device)
    kwargs = {
        "program_config": _compact_program(),
        "domain": "dense.matmul",
        "memory_config": ttnn.DRAM_MEMORY_CONFIG,
        "dtype": ttnn.bfloat16,
    }
    caller_ckc = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )
    with expect_error(ValueError, "compute_kernel_config to be omitted"):
        REGISTRY.matmul_registry_effective_default_compute_kernel_config(
            tensor_a, tensor_b, compute_kernel_config=caller_ckc, **kwargs
        )
    with expect_error(ValueError, "core_grid to be omitted"):
        REGISTRY.matmul_registry_effective_default_compute_kernel_config(
            tensor_a, tensor_b, core_grid=ttnn.CoreGrid(x=8, y=8), **kwargs
        )

    multicast = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
        in0_block_w=2,
        out_subblock_h=1,
        out_subblock_w=2,
        per_core_M=4,
        per_core_N=16,
        transpose_mcast=False,
    )
    with expect_error(ValueError, "only MatmulMultiCoreReuseProgramConfig"):
        REGISTRY.matmul_registry_effective_default_compute_kernel_config(
            tensor_a, tensor_b, **{**kwargs, "program_config": multicast}
        )
