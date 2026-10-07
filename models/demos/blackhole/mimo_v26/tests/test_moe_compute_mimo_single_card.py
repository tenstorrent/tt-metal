# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Phase A: single-card ttnn.experimental.moe_compute at MiMo dims (K=hidden=4096, N=2048, 8 experts per chip, top-8)
with bfp8 weights. Reuses the upstream single-card harness (same goldens/validators); two changes via monkeypatch:
  * weights uploaded as bfloat8_b (upstream hard-codes bfloat4_b),
  * matmul/combine PCC threshold forced to 0.999 (upstream relaxes to 0.984 for bfp4). The goldens use the bf16 weights,
    unclamped SiLU(x@w0)*(x@w1) @ w2 (MiMo has no swiglu limit), so bfp8 rounding is inside the gate.
Pass = all upstream validators + PCC >= 0.999. If the op rejects these dims (ring tile counts) the failure message is the
evidence: log it, do not weaken the gate.
"""
import os

import pytest
from ttnn.experimental.moe_compute_utils import auto_output_width_shard_dim, effective_matmul_ring_size
from ttnn.operations.ccl import MoEActivationFunction

import tests.ttnn.nightly.unit_tests.operations.experimental.test_moe_compute_single_card as up
import ttnn

PCC = 0.999


def _upload_bfp8(
    mesh_device,
    torch_w0,
    torch_w1,
    torch_w2,
    torch_b0,
    torch_b1,
    torch_b2,
    num_layers,
    experts_per_device,
    hidden_size,
    N,
    has_bias,
    w0_w1_shard_map,
    w2_shard_map,
    w0_w1_mem_config,
    w2_mem_config,
):
    assert not has_bias, "MiMo experts have no bias"
    w01 = up.prepare_w0_w1_tensor_for_moe_compute(
        torch_w0, torch_w1, num_layers, experts_per_device, hidden_size, N, w0_w1_shard_map
    )
    w2 = up.prepare_w2_tensor_for_moe_compute(
        torch_w2, num_layers, experts_per_device, N, hidden_size, w2_shard_map, w0_w1_shard_map
    )

    def put(t, mc):
        return ttnn.from_torch(
            t,
            dtype=ttnn.bfloat8_b,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=mc,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    return put(w01, w0_w1_mem_config), put(w2, w2_mem_config)


def _strict(fn):
    def wrapped(*a, **kw):
        a = list(a)
        # base_pcc_threshold is the 13th positional arg of validate_matmul; pcc_threshold kw of validate_combine
        if "pcc_threshold" in kw:
            kw["pcc_threshold"] = PCC
        elif len(a) > 12:
            a[12] = PCC
        return fn(*a, **kw)

    return wrapped


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 16384, "dispatch_core_axis": ttnn.DispatchCoreAxis.ROW, "trace_region_size": 500000}],
    indirect=True,
)
@pytest.mark.parametrize("compute_only", [True, False], ids=["compute_only", "fused_local"])
@pytest.mark.parametrize("k", [8, 4], ids=["top8", "top4"])
@pytest.mark.parametrize("mesh_shape, mesh_device", [((1, 1), (1, 1))], indirect=["mesh_device"])
def test_moe_compute_mimo_single_card(mesh_device, mesh_shape, compute_only, k, monkeypatch):
    if (
        os.environ.get("MIMO_CONTROL") != "bfp4"
    ):  # control run: unmodified upstream bfp4 upload + thresholds at MiMo dims
        monkeypatch.setattr(up, "_build_quantized_weight_tensors_cpu_prepare", _upload_bfp8)
        monkeypatch.setattr(up, "validate_matmul", _strict(up.validate_matmul))
        monkeypatch.setattr(up, "validate_combine", _strict(up.validate_combine))
    hidden = 4096
    ring_n = effective_matmul_ring_size(mesh_device)
    print(f"RESULT ring_n={ring_n} width_shard={auto_output_width_shard_dim(hidden, matmul_ring_size=ring_n)}")
    up._run_moe_compute_single_card_test(
        mesh_device=mesh_device,
        mesh_shape=mesh_shape,
        experts_per_device=8,
        tokens_per_device=32,
        selected_experts_k=k,
        N=2048,
        hidden_size=hidden,
        output_height_shard_dim=4,
        output_width_shard_dim=auto_output_width_shard_dim(hidden, matmul_ring_size=ring_n),
        dtype=ttnn.bfloat16,
        activation_type=MoEActivationFunction.SILU,
        has_bias=False,
        compute_only=compute_only,
    )
